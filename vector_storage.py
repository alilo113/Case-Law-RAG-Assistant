import os

from dotenv import load_dotenv
from pinecone import Pinecone, UpsertRecordsResponse

from chunker import chunk_pdf_texts


def initialize_and_upsert_vector_store() -> UpsertRecordsResponse:
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY must be set to initialize the vector store.")

    chunks = chunk_pdf_texts()
    if not chunks:
        raise ValueError("No PDF chunks were produced; nothing to upsert.")

    index_name = "case-law-index"
    pinecone = Pinecone(api_key=api_key)
    if not pinecone.has_index(index_name):
        pinecone.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"},
            },
        )

    records = [
        {"_id": f"chunk-{number}", "chunk_text": chunk.page_content}
        for number, chunk in enumerate(chunks)
    ]
    index = pinecone.Index(index_name)
    return index.upsert_records(namespace="case-law", records=records)