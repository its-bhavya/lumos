import google.generativeai as genai
import json, re
from config import GEMINI_API_KEY, GENERATION_MODEL

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

PROMPT_QUIZ = """
Generate {num_questions} quiz questions at a {difficulty} difficulty level,with {type} type of answers, suitable for university/school exams.

Rules:
- Base questions ONLY on the context below.
- Use both the structured topic/subtopic list AND the raw text.
- Make questions conceptual and answerable directly from the context.
- Do NOT add new facts.
- Return VALID JSON ONLY.

JSON Format:
[
  {{
    "question_num": 1,
    "question": "the question text",
    "topic": "main topic",
    "subtopic": "subtopic (or null)",
    "answer": "answer strictly from context"
  }}
]

Structured Topics (JSON):
{extracted}

Raw Context:
{context}

JSON Output:
"""
def clean_json_field(field:str):
            field = re.sub(r"^```(?:json)?\n?", "", field.strip())
            field = re.sub(r"\n?```$", "", field)
            return json.loads(field)

def generate_quiz_from_transcripts(transcript, topics_json, num_questions=5, difficulty="medium", type="short"):
    transcript = transcript
    topics_and_subtopics = topics_json
    prompt = PROMPT_QUIZ.format(
        num_questions=num_questions,
        context=transcript,
        extracted=json.dumps(topics_and_subtopics, indent=2),
        difficulty=difficulty,
        type=type

    )

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    raw_output = response.text.strip()
    clean_output = clean_json_field(raw_output)
    return clean_output
