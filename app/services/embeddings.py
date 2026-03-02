import cohere
from openai import OpenAI

from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

# Initialize Cohere client lazily to ensure settings are loaded
_cohere_client: cohere.ClientV2 | None = None


def get_cohere_client() -> cohere.ClientV2:
    """Get or create the Cohere client instance."""
    global _cohere_client
    if _cohere_client is None:
        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY is not set in environment variables")
        _cohere_client = cohere.ClientV2(api_key=settings.cohere_api_key)
    return _cohere_client


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of texts using OpenAI API."""
    response = client.embeddings.create(
        input=texts,
        model=settings.embedding_model,
    )
    return [item.embedding for item in response.data]


def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]


def rerank(query: str, documents: list[dict], top_n: int) -> list[dict]:
    """Rerank search results using Cohere and return the top_n most relevant.

    Each dict in `documents` must have a "content" key. The returned list
    preserves the original dict structure with the score replaced by the
    Cohere relevance score.
    """
    if not documents:
        return documents

    top_n = min(top_n, len(documents))

    cohere_client = get_cohere_client()
    response = cohere_client.rerank(
        model=settings.rerank_model,
        query=query,
        documents=[d["content"] for d in documents],
        top_n=top_n,
    )

    reranked: list[dict] = []
    for r in response.results:
        item = dict(documents[r.index])
        item["score"] = round(r.relevance_score, 4)
        reranked.append(item)

    return reranked


def ask_llm(query: str, context_chunks: list[str]) -> str:
    """Generate an answer using retrieved context."""
    context = "\n\n---\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on the provided document context. "
                    "Only use the information from the context below. If the context doesn't contain "
                    "enough information to answer, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        return ""
    return content
