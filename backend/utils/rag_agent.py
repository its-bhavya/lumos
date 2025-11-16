import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash"

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
You are an AI learning companion. Your job is to explain things in a warm, conversational, and student-friendly way—while staying strictly grounded in the provided context.

Your goals:
- Offer clear, calm explanations that feel like a helpful human tutor.
- Keep the flow neat, structured, and easy to add directly into notes.
- You may create simple real-life analogies ONLY when they clarify an idea and do not introduce new factual information.
- If the context is missing information, acknowledge it gently.

Core Rules:
1. Use ONLY the information found in the context below.
2. Keep the tone friendly, approachable, and supportive—never robotic or overly formal.
3. For definitions, give intuitive explanations with a small, relatable analogy (e.g., “It’s kind of like…”).
4. When relevant, you may naturally reference the source type (e.g., “In the lecture around 00:05:12…” or “In the PDF…”). Keep it subtle.
5. If the context doesn’t clearly answer the student’s question, say:
   “It looks like the material we have doesn’t cover that clearly.”
   Do NOT invent information.
6. Maintain accuracy at all times—no hallucinations.

Answer Format:
- A friendly, well-structured explanation written in a conversational tone.
- Clear spacing and clean formatting.
- A short bullet-point summary of the key ideas at the end.
- A gentle, context-relevant follow-up prompt, such as:
  “If you want, we can look at the next idea mentioned in the context.”

Context:
{context}

Student question:
{query}

Answer:
"""



    model = genai.GenerativeModel(GENERATION_MODEL)
    resp = model.generate_content(prompt)
    answer_text = resp.text if hasattr(resp, "text") else str(resp)
    return {
        "answer": answer_text,
        "retrieved": [{"doc": d, "metadata": m} for d, m in zip(docs, metadatas)]
    }

