from pathlib import Path
import os

BASE_TEMP_DIR = Path("/tmp/rag_sessions")
BASE_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Models / Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")  # optional for audio transcription fallback

# Embedding model name used with Google GenAI
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash"
