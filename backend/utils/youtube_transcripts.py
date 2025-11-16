"""
YouTube transcript extractor using yt-dlp + your existing transcribe() pipeline.
This method is extremely reliable and works even when pytube or youtube_transcript_api fail.
"""

import tempfile
import shutil
from pathlib import Path
import subprocess
from audio_service import transcribe
from textwrap import wrap


def download_youtube_audio(youtube_url: str, output_dir: str) -> Path:
    """
    Downloads audio from a YouTube URL using yt-dlp.
    Returns the path to the downloaded audio file.
    """

    output_template = str(Path(output_dir) / "audio.%(ext)s")

    command = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "-o",
        output_template,
        youtube_url,
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed: {e.stderr.decode()}") from e

    # find downloaded file
    for f in Path(output_dir).glob("audio.*"):
        return f

    raise RuntimeError("Audio file not found after yt-dlp download.")


def fetch_youtube_transcript_segments(youtube_url: str):
    """
    Downloads audio using yt-dlp → transcribes using audio_service → chunks into RAG segments.
    """

    tmpdir = tempfile.mkdtemp()

    try:
        # Step 1: Download audio
        audio_path = download_youtube_audio(youtube_url, tmpdir)

        # Step 2: Transcribe using your existing pipeline
        raw_segments = transcribe(audio_path)
        if not raw_segments:
            raise RuntimeError("Transcription returned empty result.")

        full_text = raw_segments[0]["text"]

        # Step 3: Chunk the transcript for RAG
        CHUNK_SIZE = 800  # characters

        chunks = wrap(full_text, CHUNK_SIZE)

        segments = []
        for idx, chunk in enumerate(chunks):
            segments.append({
                "source_type": "youtube",
                "source_id": youtube_url,
                "start": None,
                "end": None,
                "chunk_index": idx,
                "text": chunk
            })

        return segments

    except Exception as e:
        raise RuntimeError(f"Error extracting YouTube transcript: {e}") from e

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


segments = fetch_youtube_transcript_segments("https://youtu.be/OrM7nZcxXZU")
print(len(segments))      # number of chunks
print(segments[0]["text"])  # first chunk
