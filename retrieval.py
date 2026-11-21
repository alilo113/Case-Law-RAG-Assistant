import os

from dotenv import load_dotenv
from pinecone import Hit, Pinecone


def search_similar_chunks(query_text: str, top_k: int = 3) -> list[Hit]:
    """Return the top matching chunks for a query from the case-law index."""
    if not query_text.strip():
        raise ValueError("query_text must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY must be set to search the vector store.")

    pinecone = Pinecone(api_key=api_key)
    index = pinecone.Index("case-law-index")
    response = index.search(
        namespace="case-law",
        top_k=top_k,
        inputs={"text": query_text},
        fields=["chunk_text"],
    )
    return response.result.hits