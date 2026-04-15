from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def stream_query(base_url: str, query: str, top_k: int) -> int:
    url = f"{base_url.rstrip('/')}/query"
    payload = {"query": query, "top_k": top_k, "stream": True}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            print(f"Streaming from {url}\n")
            full_text: list[str] = []

            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[raw] {line}")
                    continue

                event_type = event.get("type")
                if event_type == "meta":
                    print(f"Query: {event.get('query', '')}\n")
                elif event_type == "token":
                    token = event.get("text", "")
                    if token:
                        print(token, end="", flush=True)
                        full_text.append(token)
                elif event_type == "done":
                    print("\n\n---")
                    print("Sources:")
                    for src in event.get("sources", []):
                        print(
                            f"- {src.get('source', 'unknown')} "
                            f"(chunk={src.get('chunk_id')}, score={src.get('score')})"
                        )
                    timings = event.get("timings", {})
                    if timings:
                        print("\nTimings (ms):")
                        for key in [
                            "retrieval_ms",
                            "prompt_build_ms",
                            "llm_inference_ms",
                            "postprocess_ms",
                            "total_ms",
                        ]:
                            if key in timings:
                                print(f"- {key}: {timings[key]}")
                    return 0
                elif event_type == "error":
                    print(f"\nServer error: {event.get('detail', 'unknown error')}", file=sys.stderr)
                    return 1
                else:
                    print(f"\nUnknown event: {event}")

            if full_text:
                print("\n\nStream ended without 'done' event.", file=sys.stderr)
                return 1
            print("No stream data received.", file=sys.stderr)
            return 1

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Connection error: {exc.reason}", file=sys.stderr)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume TechRAG NDJSON stream from /query.")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="TechRAG API base URL")
    args = parser.parse_args()

    raise SystemExit(stream_query(args.base_url, args.query, args.top_k))


if __name__ == "__main__":
    main()
