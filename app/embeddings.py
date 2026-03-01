from openai import OpenAI

from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of texts using OpenAI API."""
    response = client.embeddings.create(
        input=texts,
        model=settings.embedding_model,
    )
    return [item.embedding for item in response.data]


def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]


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
