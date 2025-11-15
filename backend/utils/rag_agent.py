import google.generativeai as genai
from config import EMBEDDING_MODEL, GENERATION_MODEL, GEMINI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def query_collection_and_answer(collection, query: str, n_results: int = 5):
    # embed the query
    q_emb_resp = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    q_emb = q_emb_resp["embedding"] if isinstance(q_emb_resp, dict) and "embedding" in q_emb_resp else q_emb_resp

    res = collection.query(query_embeddings=[q_emb], n_results=n_results)
    docs = res["documents"][0]
    metadatas = res["metadatas"][0]

    # Build a context that includes source metadata
    ctx_pieces = []
    for d, m in zip(docs, metadatas):
        src = m.get("source_type", "unknown")
        sid = m.get("source_id", "")
        start = m.get("start")
        end = m.get("end")
        page = m.get("page")
        meta_str = f"[{src} {sid}"
        if page:
            meta_str += f", page {page}"
        if start is not None:
            meta_str += f", {start}-{end if end else ''}s"
        meta_str += "]"
        ctx_pieces.append(f"{meta_str} {d}")

    context = "\n\n".join(ctx_pieces)

    prompt = f"""
You are an AI learning companion. Your job is to explain things in a conversational, student-friendly way while staying strictly grounded in the provided context.

Your goals:
- Provide clear, warm explanations that feel natural and approachable.
- Organize answers so they can fit directly into a student’s notes.
- You may create simple real-life examples to clarify ideas, but ONLY if they do not add new factual claims beyond the context. These examples should be illustrative and relatable (like comparing a concept to a daily activity), not factual extensions of the source material.
- If something isn’t in the context, acknowledge it gently.

Guidelines:
1. Use only information found in the context below.
2. Write in a conversational tone—like a helpful tutor—while keeping the flow neat and structured.
3. When asked for definitions, give simple, intuitive explanations with a small, relatable example (e.g., “It’s kind of like…”).
4. When helpful, weave in timestamps (converted to hh:mm:ss), source type, or other metadata in a natural, non-disruptive way (e.g., “Around 00:05:12, the speaker explains…”).
5. If the context doesn’t contain enough information, say something like:
   “It looks like the material we have doesn’t cover that clearly.”
6. Do NOT invent information that the context doesn’t support.

Answer format:
- A friendly, clear explanation written in a conversational tone, with simple real-life analogies only when appropriate.  
- Bullet-point summary of the essentials.  
- Casual prompts like  
“Would you also like to explore…?” or “If you’re curious, we can look at…”  
These must relate ONLY to the content in the context.

Context:
{context}

Student question:
{query}

Answer:"""


    model = genai.GenerativeModel(GENERATION_MODEL)
    resp = model.generate_content(prompt)
    answer_text = resp.text if hasattr(resp, "text") else str(resp)
    return {
        "answer": answer_text,
        "retrieved": [{"doc": d, "metadata": m} for d, m in zip(docs, metadatas)]
    }

