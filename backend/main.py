from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from session_manager import create_session, get_session, delete_session, session_store
from utils.youtube_transcripts import extract_video_id, download_transcript
from utils.pdf_service import load_pdf_chunks
from utils.audio_service import transcribe
from utils.text_service import chunk_plain_text
from utils.form_vector_index import build_collection_for_session
from utils.rag_agent import query_collection_and_answer
from pathlib import Path
import shutil
import tempfile

app = FastAPI(title="Multi-source RAG backend (hackathon)")

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
