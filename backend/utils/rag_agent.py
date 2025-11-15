import chromadb
from pathlib import Path
import google.generativeai as genai

CHROMA_DIR = Path("resources/vectorstore")
chroma_client = chromadb.PersistentClient(CHROMA_DIR)
collection = chroma_client.get_or_create_collection("podcast_segments")

def get_gemini_embeddings(texts):
    embeddings = []
    for text in texts:
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            embeddings.append(result["embedding"])
        except Exception as e:
            print(f"Embedding failed: {e}")
            embeddings.append([0]*768)   # correct size for embedding-004
    return embeddings

def rag_query(user_query, n_results=5):
    query_emb = get_gemini_embeddings([user_query])[0]

    res = collection.query(query_embeddings=[query_emb], n_results=n_results)

    all_docs = []
    all_timings = []

    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        all_docs.append(doc)
        all_timings.append({
            "file": meta.get("file"),
            "start": meta.get("start"),
            "end": meta.get("end")
        })
    
    combined_context = (
        "Retrieved Segments:\n" +
        "\n\n".join(f"[{t['file']} | {t['start']} - {t['end']}] {d}" 
                    for d, t in zip(all_docs, all_timings))
    )

    prompt = f"""
You are an AI educational assistant that answers student questions based strictly on the provided context.

Your goals:
- Help the student learn clearly and accurately.
- Use examples, step-by-step reasoning, and simple language when helpful.
- If the answer is not fully in the context, say that you are unsure rather than inventing information.

Guidelines:
1. Only use information in the context below.
2. Provide concise but clear explanations.
3. If asked for definitions, give short, intuitive explanations with examples.
4. Refer to timestamps if helpful.
5. If context is insufficient, say:
   "The provided material does not include enough information to answer that question."

Format:
**Answer:** main explanation  
**Key Points:** bullet summary  
**Suggested Follow-up Questions:** optional prompts for deeper learning, make sure that they are included in the content you have.

Context:
{combined_context}

Student question:
{user_query}

Answer:
""".strip()

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text, all_timings


if __name__ == "__main__":
    ans, timings = rag_query("What did they say about operating systems?", 5)
    print("\n📌 RAG Answer:\n", ans)
    print("\n🕒 Segments Used:\n", timings)
