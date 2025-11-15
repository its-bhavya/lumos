import google.generativeai as genai
from config import GEMINI_API_KEY, GENERATION_MODEL

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

PROMPT_NOTES = """
You are an AI learning companion.

Write clear, structured study notes based ONLY on the context below.
Use a conversational tone, but keep the flow organized and helpful.
Include short real-life analogies when useful, but do not add factual
information that isn’t already in the context.
Return the notes in a markdown format.
Context:
{context}

Notes:
"""

def generate_notes_from_transcripts(text):
    text = text

    prompt = PROMPT_NOTES.format(context=text)

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text
