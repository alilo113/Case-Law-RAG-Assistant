from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

import pdfplumber

def extract_pdf_texts() -> list[str]:
    data_dir = Path(__file__).resolve().parent / "data"
    pdf_paths = sorted(data_dir.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDF files found in {data_dir}")
        return []

    extracted_texts: list[str] = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages).strip()

        extracted_texts.append(text)

    return extracted_texts


def chunk_pdf_texts() -> list[str]:
    """Split extracted PDF texts into overlapping chunks."""
    pdf_texts = extract_pdf_texts()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents(pdf_texts)

print(chunk_pdf_texts())