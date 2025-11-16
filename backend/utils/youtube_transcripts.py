from typing import Dict, List
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
from urllib.parse import urlparse, parse_qs
import requests
import os

API_KEY = os.getenv("YOUTUBE_API_KEY")

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.query:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    if parsed.netloc and "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    if "/embed/" in parsed.path:
        return parsed.path.split("/embed/")[1]
    raise ValueError("Could not extract video id from url")

def download_transcript(video_id: str) -> Dict:
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.list(video_id)
    transcript = transcript_list.find_transcript(['en','hi']).fetch()
    formatter = JSONFormatter()
    timestamped = formatter.format_transcript(transcript)

    metadata = {}
    if API_KEY:
        url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={API_KEY}&part=snippet,statistics"
        try:
            resp = requests.get(url, timeout=10).json()
            if resp.get("items"):
                item = resp["items"][0]
                metadata = {
                    "title": item["snippet"]["title"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                    "description": item["snippet"]["description"]
                }
        except Exception:
            metadata = {}

    return {
        "source_type": "youtube",
        "source_id": video_id,
        "raw_fragments": transcript,           # list of {text, start, duration}
        "timestamped": timestamped,
        "metadata": metadata
    }
