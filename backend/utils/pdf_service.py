from pypdf import PdfReader
from typing import List, Dict

def load_pdf_chunks(file_path, chunk_size=800):
    reader = PdfReader(file_path)
    chunks = []
    chunk_index = 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # naive chunking by characters:
        pos = 0
        while pos < len(text):
            piece = text[pos: pos + chunk_size]
            chunks.append({
                "source_type": "pdf",
                "source_id": file_path.name,
                "page": i + 1,
                "chunk_index": chunk_index,
                "text": piece.strip()
            })
            chunk_index += 1
            pos += chunk_size
    return chunks
