import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

PROMPT_NOTES = """
You are an AI study assistant tasked with creating **exam-ready, structured, and comprehensive study notes**.

Instructions:
- Base notes **ONLY on the context below**. Do not introduce any information not present in the context.
- Seamlessly combine knowledge from multiple sources if provided, to make the notes coherent and unified.
- Organize notes into **logical sections and subsections**, following the topic and subtopic hierarchy when possible.
- Use **clear headings, bullet points, tables, and diagrams (if relevant)** to make the notes easy to read and revise.
- Keep explanations concise but comprehensive; highlight **key definitions, formulas, examples, and important points**.
- Include **short real-life analogies or examples** only to aid understanding.
- Avoid casual language; maintain a **professional and academic tone** suitable for university/school exams.
- Notes should be structured for **easy revision and comprehension**.

Return the notes in **Markdown format**.

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
