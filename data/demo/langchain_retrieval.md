4/15/26, 11:23 AM

Retrieval - Docs by LangChain

Join us May 13th & May 14th at Interrupt, the Agent Conference by LangChain. Buy tickets >

Advanced usage

Retrieval

Advanced usage

Retrieval

Copy page

Large Language Models (LLMs) are powerful, but they have two key limitations:

Finite context—they canʼt ingest entire corpora at once.

Sttic knowledge—their training data is frozen at a point in time.

Retrieval addresses these problems by fetching relevant external knowledge at query

time. This is the foundation of Retrievl-Augmented Genertion (RAG): enhancing an
LLMʼs answers with context-specific information.

Building a knowledge base

A knowledge bse is a repository of documents or structured data used during retrieval.

If you need a custom knowledge base, you can use LangChainʼs document loaders and

vector stores to build one from your own data.

If you already have a knowledge base (e.g., a SQL database, CRM, or internal
documentation system), you do not need to rebuild it. You can:

Connect it as a tool for an agent in Agentic RAG.

Query it and supply the retrieved content as context to the LLM

(2-Step RAG)

.

See the following tutorial to build a searchable knowledge base and minimal RAG

workflow:

https://docs.langchain.com/oss/python/langchain/retrieval

1/8

4/15/26, 11:23 AM

Retrieval - Docs by LangChain

Tutorial: Semantic search

Join us May 13th & May 14th at Interrupt, the Agent Conference by LangChain. Buy tickets >

Learn how to create a searchable knowledge base from your own data using

LangChainʼs document loaders, embeddings, and vector stores. In this tutorial,

youʼll build a search engine over a PDF, enabling retrieval of passages relevant to a

query. Youʼll also implement a minimal RAG workflow on top of this engine to see

how external knowledge can be integrated into LLM reasoning.

Learn more

From retrieval to RAG

Retrieval allows LLMs to access relevant context at runtime. But most real-world

applications go one step further: they integrte retrievl with genertion to produce
grounded, context-aware answers.

This is the core idea behind Retrievl-Augmented Genertion (RAG). The retrieval
pipeline becomes a foundation for a broader system that combines search with

generation.

Retrieval pipeline

A typical retrieval workflow looks like this:

Sources

(Google Drive, Slack, Notion,

Document Loaders

Documents

Split into chunks

Turn into embeddings

User Query

Query embedding

Vector Store

Retriever

LLM uses retrieved info

Answer

Each component is modular: you can swap loaders, splitters, embeddings, or vector

stores without rewriting the appʼs logic.

Building blocks

Document loaders

Text splitters

Ingest data from external sources
(Google Drive, Slack, Notion, etc.),

Break large docs into smaller chunks
that will be retrievable individually

https://docs.langchain.com/oss/python/langchain/retrieval

2/8

4/15/26, 11:23 AM

Retrieval - Docs by LangChain

returning standardized

Join us May 13th & May 14th at Interrupt, the Agent Conference by LangChain. Buy tickets >

Document

and fit within a modelʼs context

objects.

Learn more

window.

Learn more

Embedding models

Vector stores

An embedding model turns text into

Specialized databases for storing

a vector of numbers so that texts

and searching embeddings.

with similar meaning land close

together in that vector space.

Learn more

Learn more

Retrievers

A retriever is an interface that
returns documents given an

unstructured query.

Learn more

RAG architectures

RAG can be implemented in multiple ways, depending on your systemʼs needs. We outline

each type in the sections below.

Architecture

Description

Control

Flexibility

Latency

2-Step RAG

Retrieval always

happens before

generation. Simple

and predictable

✅  High

❌  Low

⚡  Fast

https://docs.langchain.com/oss/python/langchain/retrieval

3/8

4/15/26, 11:23 AM

Retrieval - Docs by LangChain

Architecture

Control
Join us May 13th & May 14th at Interrupt, the Agent Conference by LangChain. Buy tickets >

Description

Flexibility

Latency

Agentic RAG

An LLM-powered

agent decides when

and how to retrieve

during reasoning

Combines

characteristics of

❌  Low

✅  High

⏳  Variable

Hybrid

both approaches

⚖  Medium

⚖  Medium

⏳  Variable

with validation

steps

Ltency: Latency is generally more predictble in 2-Step RAG, as the maximum number of
LLM calls is known and capped. This predictability assumes that LLM inference time is the
dominant factor. However, real-world latency may also be affected by the performance of
retrieval steps—such as API response times, network delays, or database queries—which
can vary based on the tools and infrastructure in use.

2-step RAG

In 2-Step RAG, the retrieval step is always executed before the generation step. This
architecture is straightforward and predictable, making it suitable for many applications

where the retrieval of relevant documents is a clear prerequisite for generating an

answer.

User Question

Retrieve Relevant Documents

Generate Answer

Return Answer to User

Tutorial: Retrieval-Augmented Generation (RAG)

See how to build a Q&A chatbot that can answer questions grounded in your data
using Retrieval-Augmented Generation. This tutorial walks through two

approaches:

A RAG gent that runs searches with a flexible tool—great for general-
purpose use.

A 2-step RAG chain that requires just one LLM call per query—fast and
efficient for simpler tasks.

https://docs.langchain.com/oss/python/langchain/retrieval

4/8

4/15/26, 11:23 AM

Retrieval - Docs by LangChain

Learn more

Join us May 13th & May 14th at Interrupt, the Agent Conference by LangChain. Buy tickets >

Agentic RAG

Agentic Retrievl-Augmented Genertion (RAG) combines the strengths of Retrieval-
Augmented Generation with agent-based reasoning. Instead of retrieving documents

before answering, an agent (powered by an LLM) reasons step-by-step and decides

when and how to retrieve information during the interaction.

The only thing an agent needs to enable RAG behavior is access to one or more tools that
can fetch external knowledge—such as documentation loaders, web APIs, or database
queries.

User Input / Question

Agent (LLM)

Enough to answer?

Yes

Need external info?

Yes

Search using tool(s)

No

Generate final answer

Return to user

No

https://docs.langchain.com/oss/python/langchain/retrieval

5/8

4/15/26, 11:23 AM

Retrieval - Docs by LangChain

Join us May 13th & May 14th at Interrupt, the Agent Conference by LangChain. Buy tickets >

import requests

from langchain.tools import tool

from langchain.chat_models import init_chat_model

from langchain.agents import create_agent

@tool

def fetch_url(url: str) -> str:

    """Fetch text content from a URL"""

    response = requests.get(url, timeout=10.0)

    response.raise_for_status()

    return response.text

system_prompt = """\

Use fetch_url when you need to fetch information from a web-page; quote relevan

"""

agent = create_agent(

    model="claude-sonnet-4-6",

    tools=[fetch_url], # A tool for retrieval

    system_prompt=system_prompt,

)

Show Extended example: Agentic RAG for LangGraph's llms.txt

Tutorial: Retrieval-Augmented Generation (RAG)

See how to build a Q&A chatbot that can answer questions grounded in your data
using Retrieval-Augmented Generation. This tutorial walks through two

approaches:

A RAG gent that runs searches with a flexible tool—great for general-
purpose use.

A 2-step RAG chain that requires just one LLM call per query—fast and
efficient for simpler tasks.

Learn more

https://docs.langchain.com/oss/python/langchain/retrieval

6/8

4/15/26, 11:23 AM

Retrieval - Docs by LangChain

Hybrid RAG

Join us May 13th & May 14th at Interrupt, the Agent Conference by LangChain. Buy tickets >

Hybrid RAG combines characteristics of both 2-Step and Agentic RAG. It introduces

intermediate steps such as query preprocessing, retrieval validation, and post-generation

checks. These systems offer more flexibility than fixed pipelines while maintaining some

control over execution.

Typical components include:

Query enhncement: Modify the input question to improve retrieval quality. This can
involve rewriting unclear queries, generating multiple variations, or expanding

queries with additional context.

Retrievl vlidtion: Evaluate whether retrieved documents are relevant and
sufficient. If not, the system may refine the query and retrieve again.

Answer vlidtion: Check the generated answer for accuracy, completeness, and
alignment with source content. If needed, the system can regenerate or revise the

answer.

The architecture often supports multiple iterations between these steps:

User Question

Query Enhancement

Retrieve Documents

Yes

Generate Answer

Answer Quality OK?

No

Try Different Approach?

Yes

Refine Query

Sufficient Info?

No

Yes

No

Return Best Answer

Return to User

This architecture is suitable for:

Applications with ambiguous or underspecified queries

Systems that require validation or quality control steps

Workflows involving multiple sources or iterative refinement

Tutorial: Agentic RAG with Self-Correction
An example of Hybrid RAG that combines agentic reasoning with retrieval and self-
correction.

https://docs.langchain.com/oss/python/langchain/retrieval

7/8

4/15/26, 11:23 AM

Retrieval - Docs by LangChain

Learn more

Join us May 13th & May 14th at Interrupt, the Agent Conference by LangChain. Buy tickets >

Edit this page on GitHub

 or

file an issue

.

Connect these docs

 to Claude, VSCode, and more via MCP for real-time answers.

Was this page helpful?

Yes

No

Resources

Forum

Changelog

LangChain Academy

Contact Sales

Company

Home

Trust Center

Careers

Blog

https://docs.langchain.com/oss/python/langchain/retrieval

8/8

