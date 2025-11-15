from typing import Dict
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter, JSONFormatter
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

import os, requests, dotenv, json, isodate
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# Load environment variables
dotenv.load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")

# Set up paths
CURRENT_DIR = Path(__file__).resolve().parent
TXT_TRANSCRIPT_DIR = CURRENT_DIR.parent.parent / "resources" / "transcripts" / "txt"
JSON_TRANSCRIPT_DIR = CURRENT_DIR.parent.parent / "resources" / "transcripts" / "json"
METADATA_DIR = CURRENT_DIR.parent.parent / "resources" / "metadata"
ytt_api = YouTubeTranscriptApi()

class YouTubeTranscriptDownloader:
    text_formatter = TextFormatter()
    json_formatter = JSONFormatter()

    def get_video_metadata(self, video_id: str) -> Dict:
        url = f'https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet,statistics'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data and len(data['items']) > 0:
                video_data = data['items'][0]
                return {
                    'title': video_data['snippet']['title'],
                    'channel_title': video_data['snippet']['channelTitle'],
                    'published_at': video_data['snippet']['publishedAt'],
                    'view_count': video_data['statistics']['viewCount'],
                    'like_count': video_data['statistics'].get('likeCount', 0),
                    'comment_count': video_data['statistics'].get('commentCount', 0),
                    'description': video_data['snippet']['description']
                }
        return {}

    def download_transcript(self, video_id: str) -> Dict:
        try:
            print(f"Processing video_id: {video_id}")
            transcript_list = ytt_api.list(video_id)
            transcript = transcript_list.find_transcript(['en'])
            transcript = transcript.fetch()

            formatted_text = self.text_formatter.format_transcript(transcript)
            formatted_json = self.json_formatter.format_transcript(transcript)

            metadata = self.get_video_metadata(video_id)

            return {
                'video_id': video_id,
                'transcript': formatted_text,
                'raw_transcript': transcript,
                'timestamped_transcript': formatted_json,
                'metadata': metadata
            }

        except NoTranscriptFound:
            raise ValueError(f"No transcript found for video: {video_id}")
        except TranscriptsDisabled:
            raise ValueError(f"Transcripts are disabled for video: {video_id}")
        except Exception as e:
            raise Exception(f"Error downloading transcript: {str(e)}")


def extract_video_id_from_url(url: str) -> str:
    """Extract video ID from any YouTube URL."""
    parsed = urlparse(url)

    # Standard watch?v= URL
    if parsed.query:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

    # Short youtu.be/xxxx URL
    if parsed.netloc in ["youtu.be"]:
        return parsed.path.lstrip("/")

    # Embed URLs
    if "/embed/" in parsed.path:
        return parsed.path.split("/embed/")[1]

    raise ValueError("Could not extract video ID from URL")

# =============== MAIN USAGE ===============

if __name__ == "__main__":
    yt_url = input("Enter YouTube Video URL: ").strip()
    try:
        video_id = extract_video_id_from_url(yt_url)
        downloader = YouTubeTranscriptDownloader()
        result = downloader.download_transcript(video_id)

        # Determine filename
        published_at = result['metadata'].get('published_at')
        if published_at:
            publish_date = datetime.fromisoformat(published_at.replace("Z", "")).strftime("%d%m%Y")
            base_name = publish_date
        else:
            base_name = result['video_id']

        # Ensure directories exist
        TXT_TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        JSON_TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)

        # Paths
        txt_transcript_path = TXT_TRANSCRIPT_DIR / f"{base_name}.txt"
        json_transcript_path = JSON_TRANSCRIPT_DIR / f"{base_name}.json"
        metadata_path = METADATA_DIR / f"{base_name}.json"

        # Save files
        with open(txt_transcript_path, "w", encoding="utf-8") as f:
            f.write(result['transcript'])

        with open(json_transcript_path, "w", encoding="utf-8") as f:
            json.dump(json.loads(result['timestamped_transcript']), f, indent=2, ensure_ascii=False)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(result['metadata'], f, indent=2, ensure_ascii=False)

        print(f"Saved transcript and metadata as {base_name}")

    except Exception as e:
        print(f"Error: {e}")
