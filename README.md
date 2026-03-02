Hello, this is a demo project to display pdf RAG+retrieval capabilities.

We have 38 pdfs in example-pdfs repo, all about current geopolitics.

Our system consist of these steps:

1) make ingest -> OCRs the pdfs, chunk them, and embed those chunks.
2) make enrich -> Enrich the pdf data with llm generated memory
3) make test-questions -> Generate random question each related to a single pdf and chunk.
4) make search-eval -> Asks the generated questions and measure the system's accuracy.

Bonus: "make chat" for chat cli do converse with the documents.

# STACK

We use uv with python 3.11. To run, simply install "uv" (https://github.com/astral-sh/uv) in your system, then run these commands:

$ make deps
$ make build
$ make dev -> server is running


And you are set.


# DESIGN

## Embedding time

So to make the case more interesting, I employed a dynamical memory, which for each document, allows the LLM to extract memories out of the chunks it traverse.

I asked llm to keep track of a short term memory as part of its context. It is limited, and we remove the oldest entries every turn, the dropped entries goes into the long term memory.

Each memory is either a "context" of a "fact". We also use these memories to generate a description and a list of tags for the document.

## Retrieval Time

During the retrieval, we first take user query, and expand it further into queries and regexes more appropriate to run on our data model, mainly memory contexts and facts. We also ask LLM to attach a "relevancy score" to each query or regex it generated.

Then we employ a scoring algorithm to bring all those different kind of matching together, and come up with a final document score, which we display to user.

## DATA

As an example data, I manually downloaded around 40 contemporary geopolitics related PDFs from the web, all under 20 pages. I also have a WIP selenium progress crawler in scripts/download_pdfs.py which I could not finist.

## RESULT

Each query takes around 8-10 seconds, using openai API for the llm.


## UI

After you run make dev, hit `http://localhost:8000` to view the ui. You can search, chat, inspect documents or generated test questions.

## TODO

- We can add paddleOCR to better parse the layout, helping in both chunking and detecting tables/figures etc.
-- I tried it, but took forever on a CPU, so we can try optimize that or go for a GPU based solution.
- We can also use multi-modal-LLM based processing for layout, more expensive yes, but when you think about it we pass whole data trough the LLM once for metadata enrichment anyway.
- Later, we can introduce a "chunk type", to make tables and figures first class citizens in our system and handle them specially.
- We can skip enrichment if showing the whole PDF data to the LLM is too expensive.
- And ofc, we can improve our eval script to better confirm with user use cases.