import chromadb
import google.generativeai as genai
from config import EMBEDDING_MODEL, GEMINI_API_KEY
from typing import List

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------------------------------------
# FAST BATCH EMBEDDING
# ----------------------------------------------------------
def embed_texts(texts: List[str], batch_size=32):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document"
            )

            # gemini returns a list of embedding dicts
            batch_embs = resp["embedding"]
            embeddings.extend(batch_embs)

        except Exception as e:
            print("Embedding failed:", e)
            # fallback zero vector
            for _ in batch:
                embeddings.append([0]*768)

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

    # extract texts
    texts = [s["text"] for s in segments if s.get("text")]
    if not texts:
        return client, collection

    # fast embedding
    embeddings = embed_texts(texts)

    # aligned metadata list
    metadatas = []
    for s in segments:
        if not s.get("text"):
            continue

        md = {}
        for k in ("source_type", "source_id", "start", "end", "page", "chunk_index"):
            if k in s:
                md[k] = s[k]

        metadatas.append(md)

    # aligned ids
    ids = [str(i) for i in range(len(texts))]

    # add everything in one fast call
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return client, collection
