import uuid
import shutil
from pathlib import Path
from config import BASE_TEMP_DIR

# In-memory session store:
# session_store[session_id] = {
#   "segments": [ {text, source_type, source_id, start, end, page, chunk_index} ],
#   "collection": chroma_collection_obj,
#   "client": chroma_client_obj,
#   "metadata": { ... }
# }
session_store = {}

def create_session():
    session_id = str(uuid.uuid4())
    session_dir = BASE_TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    session_store[session_id] = {
        "segments": [],
        "collection": None,
        "client": None,
        "metadata": {}
    }
    return session_id

def get_session(session_id):
    return session_store.get(session_id)

def delete_session(session_id):
    sess = session_store.pop(session_id, None)
    path = BASE_TEMP_DIR / session_id
    if path.exists():
        shutil.rmtree(path)
    return bool(sess)
