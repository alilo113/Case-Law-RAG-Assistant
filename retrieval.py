import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
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


def _format_hits_for_prompt(hits: list[Hit]) -> str:
    """Convert Pinecone Hit objects into labeled context blocks."""
    blocks: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        fields = hit.get("fields", {}) if isinstance(hit, dict) else getattr(hit, "fields", {}) or {}
        hit_id = hit.get("_id", f"chunk_{idx}") if isinstance(hit, dict) else getattr(hit, "_id", f"chunk_{idx}")
        chunk_text = fields.get("chunk_text", "").strip()
        if chunk_text:
            blocks.append(f"[Source {idx} | ID: {hit_id}]\n{chunk_text}")
    return "\n\n".join(blocks)

def answer_case_law_question(question: str, top_k: int = 3) -> str:
    """Retrieve case-law chunks from Pinecone and generate a grounded answer via Gemini."""
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY must be set to generate answers.")

    hits = search_similar_chunks(question, top_k=top_k)
    context = _format_hits_for_prompt(hits)

    if not context:
        return "Insufficient evidence in indexed sources to answer this question."

    client = genai.Client(api_key=gemini_key)

    system_instruction = (
        "You are a legal research assistant. Answer ONLY using the provided SOURCES. "
        "Cite every claim using [Source X] matching the provided source blocks. "
        "Do not use outside legal knowledge, assume missing facts, or invent citations. "
        "If the SOURCES do not contain enough information to answer, reply exactly: "
        "'Insufficient evidence in indexed sources to answer this question.'"
    )

    prompt = f"""SOURCES:
{context}

QUESTION:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
        ),
    )
    print("HELLO WORLD")
    return (response.text or "").strip()


if __name__ == "__main__":
    sample_q = "What is the case name, appellate case number, and date of this decision?"
    print(answer_case_law_question(sample_q, top_k=3))