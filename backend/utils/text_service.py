from typing import List, Dict

def chunk_plain_text(text: str, source_id: str = "inline_text", chunk_chars=800):
    chunks = []
    pos = 0
    idx = 0
    while pos < len(text):
        piece = text[pos:pos + chunk_chars]
        chunks.append({
            "source_type": "text",
            "source_id": source_id,
            "chunk_index": idx,
            "text": piece.strip()
        })
        pos += chunk_chars
        idx += 1
    return chunks
