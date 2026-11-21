from pathlib import Path

import pdfplumber


def extract_pdf_texts() -> dict[str, str]:
    """Extract and print the text of every PDF in the project's data directory."""
    data_dir = Path(__file__).resolve().parent / "data"
    pdf_paths = sorted(data_dir.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDF files found in {data_dir}")
        return {}

    extracted_texts = {}
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages).strip()

        extracted_texts[pdf_path.name] = text
        print(f"\n--- {pdf_path.name} ---")
        print(text)

    return extracted_texts


if __name__ == "__main__":
    extract_pdf_texts()
