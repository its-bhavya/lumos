import chromadb, os
import google.generativeai as genai
from typing import List
from concurrent.futures import ThreadPoolExecutor

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"

# ----------------------------------------------------------
# FAST BATCH EMBEDDING WITH PARALLELIZATION
# ----------------------------------------------------------
def embed_texts(texts: List[str], batch_size=12, max_workers=10):
    embeddings = []

    def embed_batch(batch):
        try:
            resp = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document"
            )
            # gemini returns a list of embedding dicts
            return resp["embedding"]
        except Exception as e:
            print("Embedding failed:", e)
            return [[0]*768 for _ in batch]

    # Create batches
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    # Parallel embedding
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(embed_batch, batches)

    for batch_emb in results:
        embeddings.extend(batch_emb)

    return embeddings


# ----------------------------------------------------------
# BUILD FAST COLLECTION
# ----------------------------------------------------------
def build_collection_for_session(session_id: str, segments: List[dict]):
    client = chromadb.Client()
    col_name = f"session_{session_id}"

    # reset collection
    try:
        client.delete_collection(col_name)
    except:
        pass

    collection = client.create_collection(name=col_name)

    # Pre-filter segments and align metadata in a single pass
    texts, metadatas = [], []
    for s in segments:
        t = s.get("text")
        if t:
            texts.append(t)
            md = {k: s[k] for k in ("source_type", "source_id", "start", "end", "page", "chunk_index") if k in s}
            metadatas.append(md)

    if not texts:
        return client, collection

    # Parallel embedding
    embeddings = embed_texts(texts)

    # aligned ids
    ids = [str(i) for i in range(len(texts))]

    BATCH_ADD_SIZE = 1000  # safe batch size for Chroma

    for i in range(0, len(texts), BATCH_ADD_SIZE):
        batch_ids = ids[i:i+BATCH_ADD_SIZE]
        batch_texts = texts[i:i+BATCH_ADD_SIZE]
        batch_embeddings = embeddings[i:i+BATCH_ADD_SIZE]
        batch_metadatas = metadatas[i:i+BATCH_ADD_SIZE]

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )
    return client, collection
