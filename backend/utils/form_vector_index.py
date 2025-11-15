import json
import chromadb
from pathlib import Path
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Path to your processed segments
SEGMENT_DIR = Path("resources/transcripts/processed/segments")

# ChromaDB persistent directory
CHROMA_DIR = Path("resources/vectorstore")
CHROMA_DIR.mkdir(exist_ok=True)


def embed_texts(texts: List[str], batch_size=10) -> List[List[float]]:
    """
    Generate embeddings using Gemini embedding model in batches.
    Shows progress with tqdm.
    """
    model = "models/text-embedding-004"
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        embeddings = genai.embed_content(
            model=model,
            content=batch,
            task_type="retrieval_document"
        )
        all_embeddings.extend(embeddings['embedding'])

    return all_embeddings


def build_vector_index():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="podcast_segments",
        metadata={"hnsw:space": "cosine"}
    )

    segment_files = list(SEGMENT_DIR.glob("*.json"))
    print(f"Found {len(segment_files)} segment files.\n")

    # tqdm progress for JSON files
    for seg_file in tqdm(segment_files, desc="Processing JSON files"):
        with open(seg_file, "r", encoding="utf-8") as f:
            segments = json.load(f)

        texts = [seg["text"] for seg in segments]
        ids = [f"{seg_file.stem}_{i}" for i in range(len(segments))]

        # Batch-embedding with progress bar
        embeddings = embed_texts(texts)

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"file": seg_file.stem, "start": seg["start"], "end": seg["end"]}
                for seg in segments
            ]
        )

    print("\n✨ Vector index build complete!")


if __name__ == "__main__":
    build_vector_index()
