from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
from session_manager import create_session, get_session, delete_session, session_store
from config import GEMINI_API_KEY
from utils.youtube_transcripts import extract_video_id, download_transcript
from utils.pdf_service import load_pdf_chunks
from utils.audio_service import transcribe
from utils.text_service import chunk_plain_text
from utils.form_vector_index import build_collection_for_session
from utils.rag_agent import query_collection_and_answer
from generators.generate_notes import generate_notes_from_transcripts
from generators.generate_quiz import generate_quiz_from_transcripts
from generators.extractor_agent import MindmapExtractor, KeyPointExtractor, clean_json_field
from generators.mindmap_generator import generate_mindmap_svg_from_json
from generators.evaluation import AnswerEvaluator
from pathlib import Path
import shutil
import tempfile
import dspy
import json
import re

api_key = GEMINI_API_KEY
app = FastAPI(title="Multi-source RAG backend (hackathon)")
dspy.configure(lm=dspy.LM("gemini/gemini-2.0-flash", api_key=api_key))

@app.post("/create_session")
def api_create_session():
    sid = create_session()
    return {"session_id": sid}

@app.post("/add_youtube")
def api_add_youtube(session_id: str = Form(...), youtube_url: str = Form(...)):
    sess = get_session(session_id)
    if not sess:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)
    try:
        vid = extract_video_id(youtube_url)
        data = download_transcript(vid)
        fragments = data["raw_fragments"]
        # Convert fragments to simple segments (merge small fragments into ~sentences)
        # simplistic: use each fragment as a segment, but you can reuse your merge/segment code
        for i, frag in enumerate(fragments):
            start = frag.start
            end = frag.start + frag.duration
            text = frag.text

            sess["segments"].append({
                "source_type": "youtube",
                "source_id": vid,
                "start": start,
                "end": end,
                "chunk_index": i,
                "text": text
            })
        return {"status": "ok", "added": len(fragments)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/upload_pdf")
async def api_upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    sess = get_session(session_id)
    if not sess:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)

    # save temporarily
    tmpdir = tempfile.mkdtemp()
    tmp_path = Path(tmpdir) / file.filename
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        chunks = load_pdf_chunks(tmp_path)
        sess["segments"].extend(chunks)
        return {"status": "ok", "added": len(chunks)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.post("/upload_audio")
async def api_upload_audio(session_id: str = Form(...), file: UploadFile = File(...)):
    sess = get_session(session_id)
    if not sess:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)
    tmpdir = tempfile.mkdtemp()
    tmp_path = Path(tmpdir) / file.filename
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        segments = transcribe(tmp_path)
        sess["segments"].extend(segments)
        return {"status": "ok", "added": len(segments)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.post("/add_text")
def api_add_text(session_id: str = Form(...), source_name: str = Form("inline"), text: str = Form(...)):
    sess = get_session(session_id)
    if not sess:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)
    chunks = chunk_plain_text(text, source_id=source_name)
    sess["segments"].extend(chunks)
    return {"status": "ok", "added": len(chunks)}

@app.post("/build_index")
def api_build_index(session_id: str = Form(...)):
    sess = get_session(session_id)
    if not sess:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)
    # if collection already exists, rebuild from scratch (fresh start for new upload)
    client, collection = build_collection_for_session(session_id, sess["segments"])
    sess["client"] = client
    sess["collection"] = collection
    return {"status": "ok", "segments": len(sess["segments"])}

@app.post("/ask")
def api_ask(session_id: str = Form(...), question: str = Form(...)):
    sess = get_session(session_id)
    if not sess or not sess.get("collection"):
        return JSONResponse({"error": "invalid session_id or index not built"}, status_code=400)
    out = query_collection_and_answer(sess["collection"], question, n_results=5)
    return out

@app.post("/reset")
def api_reset(session_id: str = Form(...)):
    ok = delete_session(session_id)
    return {"deleted": ok}

@app.post("/get_notes")
def api_get_notes(session_id: str = Form(...)):
    sess = get_session(session_id)
    if not sess:
        return JSONResponse({"error":"invalid session_id"}, status_code=400)
    
    if "segments" not in sess or len(sess["segments"])==0:
        return JSONResponse({"error":"No content added"}, status_code=400)
    
    docs = [s["text"] for s in sess["segments"] if s.get("text")]

    notes = generate_notes_from_transcripts(docs)
    return {"notes":notes}

@app.post("/extract")
def api_extract_topics_json(session_id: str = Form(...)):
    extractor = MindmapExtractor()
    sess = get_session(session_id)
    if not sess:
        return JSONResponse({"error":"invalid session_id"}, status_code=400)
    
    if "segments" not in sess or len(sess["segments"])==0:
        return JSONResponse({"error":"No content added"}, status_code=400)

    transcript = [s["text"] for s in sess["segments"] if s.get("text")]
    transcript = ' '.join(transcript)
    try:
        result = extractor.forward(transcript=transcript)
        def clean_json_field(field:str):
            field = re.sub(r"^```(?:json)?\n?", "", field.strip())
            field = re.sub(r"\n?```$", "", field)
            return json.loads(field)
        subtopics = clean_json_field(result.subtopics)
        data = {"central_topic":result.central_topic, "subtopics":subtopics}
        sess["extracted_topics"] = data
        return data
    except Exception as e:
        return JSONResponse({"error":"Error while extracting topics:{e}"}, status_code=500)

@app.post("/generate_mindmap")
def api_generate_mindmap(session_id: str = Form(...)):
    sess = get_session(session_id)

    if not sess or "extracted_topics" not in sess:
        return JSONResponse({"error": "Run /extract first"}, status_code=400)

    data = sess["extracted_topics"]

    
    svg_clean = generate_mindmap_svg_from_json(data)
    return Response(content=svg_clean, media_type="image/svg+xml")

@app.post("/get_quiz_questions")
def api_generate_quiz(session_id: str = Form(...), num_questions: int = Form(5), difficulty: str = Form("medium"), type:str = Form("Short")):
    sess = get_session(session_id)
    if not sess:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)

    if "segments" not in sess or not sess["segments"]:
        return JSONResponse({"error": "No content added"}, status_code=400)
    
    if "extracted_topics" not in sess:
        return JSONResponse({"error": "Please call /extract first."}, status_code=400)

    # raw text
    docs = [s["text"] for s in sess["segments"] if s.get("text")]
    full_context = "\n".join(docs)

    # structured context
    extracted = sess["extracted_topics"]

    quiz = generate_quiz_from_transcripts(
        topics_json=extracted,
        transcript=full_context,
        num_questions=num_questions,
        difficulty=difficulty,
        type=type
    )

    sess["quiz"] = quiz
    sess["quiz_answers"] = {}
    return {"quiz_questions": quiz}

@app.post("/submit_answer")
def submit_answer(session_id: str = Form(...), question_num:int = Form(...), user_answer:str = Form(...)):

    sess = get_session(session_id)
    if "quiz" not in sess:
        return JSONResponse({"error":"No quiz started"}, status_code=400)
    
    q = next((q for q in sess["quiz"] if q["question_num"]==question_num), None)
    if not q:
        return JSONResponse({"error":"invalid question_num"}, status_code=400)
    
    correct_answer = q["answer"]
    keypoints_extractor = KeyPointExtractor()
    result_raw = keypoints_extractor(answer=correct_answer)
    result = clean_json_field(result_raw.key_points)

    evaluator = AnswerEvaluator()
    evaluation_json_raw = evaluator(user_answer=user_answer, keypoints=result,  correct_answer=correct_answer)
    evaluations_json = clean_json_field(evaluation_json_raw.evaluation_json)

    sess["quiz_answers"][question_num] = {
        "user_answer":user_answer, 
        **evaluations_json
    }

    return evaluations_json

@app.post("/finish_quiz")
def finish_quiz(session_id:str = Form(...)):
    sess = get_session(session_id)

    quiz = sess.get("quiz", [])
    answers = sess.get("quiz_answers", {})

    total_q = len(quiz)
    print(total_q)
    total_score = sum(a["score"] for a in answers.values())
    print(total_score)

    accuracy = (total_score/total_q)*100

    topic_scores = {}

    for q in quiz:
        num = q['question_num']
        print(num)
        print(q['question'])
        topic = q['topic']

        score = answers.get(num, {}).get('score', 0)
        print(score)
        topic_scores.setdefault(topic, []).append(score)

    topic_strength = {
        t: (sum(scores)/len(scores))
        for t, scores in topic_scores.items()
    }

    strong = [t for t, s in topic_strength.items() if s>=0.8]
    weak = [t for t, s in topic_strength.items() if s <0.5]

    result = {
        "accuracy":round(accuracy, 1),
        "strong_topics":strong,
        "weak_topics":weak,
        "topic_strength":topic_strength
    }

    sess['quiz_result'] = result
    return result
    