from __future__ import annotations

import json
from functools import lru_cache
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import Response
from starlette.responses import StreamingResponse

from chain import TechRAG


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    stream: bool = False


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]
    timings: dict[str, float] | None = None


app = FastAPI(title="TechRAG API", version="0.1.0")


@lru_cache(maxsize=1)
def get_rag() -> TechRAG:
    return TechRAG()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {
        "service": "TechRAG API",
        "status": "ok",
        "endpoints": {
            "health": "GET /health",
            "query": "POST /query",
        },
    }


@app.get("/query")
def query_help() -> dict:
    return {
        "detail": "Use POST /query with JSON body.",
        "example": {"query": "Explain OFDM waveform design for 5G NR", "top_k": 5, "stream": True},
    }


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    rag = get_rag()
    if req.stream:
        try:
            prepared = rag.prepare_query(req.query, top_k=req.top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

        def generate():
            t0 = perf_counter()
            yield json.dumps({"type": "meta", "query": req.query}, ensure_ascii=False) + "\n"
            try:
                for token in rag.stream_from_prompt(prepared.prompt):
                    yield json.dumps({"type": "token", "text": token}, ensure_ascii=False) + "\n"
                t1 = perf_counter()
                timings = dict(prepared.timings)
                timings["llm_inference_ms"] = round((t1 - t0) * 1000, 2)
                timings["total_ms"] = round(sum(timings.values()), 2)
                yield (
                    json.dumps(
                        {
                            "type": "done",
                            "query": req.query,
                            "sources": prepared.sources,
                            "timings": timings,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            except Exception as exc:
                yield json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    try:
        result, timings = rag.ask_with_timings(req.query, top_k=req.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    return QueryResponse(**result.__dict__, timings=timings)
