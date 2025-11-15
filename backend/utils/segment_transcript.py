import json, re
from pathlib import Path

def merge_fragments_into_sentences(fragments, max_gap=1.0):
    sentences = []
    current = {"start": None, "end": None, "text": ""}
    for frag in fragments:
        text = frag["text"].strip()
        start = frag["start"]
        end = start + frag["duration"]
        if current["start"] is None:
            current["start"] = start
        gap = start - current["end"] if current["end"] is not None else 0
        current["text"] += (" " if current["text"] else "") + text
        current["end"] = end
        if re.search(r"[.?!]['\")\]]*$", text) or gap > max_gap:
            sentences.append(current)
            current = {"start": None, "end": None, "text": ""}
    if current["text"]:
        sentences.append(current)
    return sentences

def segment_by_gap_and_keywords(sentences, time_gap=2.5, keywords=None):
    if keywords is None:
        keywords = ["next", "question", "let's move", "our guest", "first off", "so let's", "now, before we", "says", "so, now", "first question", "next question", "last question", "final question"]

    segments = []
    current_segment = []
    prev_end = None

    for sent in sentences:
        gap = sent["start"] - prev_end if prev_end is not None else 0
        keyword_hit = any(re.search(re.escape(kw), sent["text"].lower()) for kw in keywords)
        says_pattern_hit = bool(re.search(r"\b[a-z]+(?: [a-z]+)? says\b", sent["text"].lower()))

        if keyword_hit:
            print(f"Keyword hit on: {sent['text']}")
        if gap > time_gap or keyword_hit or says_pattern_hit:
            if current_segment:
                segment = {
                    "start": current_segment[0]["start"],
                    "end": current_segment[-1]["end"],
                    "text": " ".join(s["text"] for s in current_segment)
                }
                segments.append(segment)
            current_segment = [sent]
        else:
            current_segment.append(sent)

        prev_end = sent["end"]

    # Add the final segment
    if current_segment:
        segment = {
            "start": current_segment[0]["start"],
            "end": current_segment[-1]["end"],
            "text": " ".join(s["text"] for s in current_segment)
        }
        segments.append(segment)

    return segments


if __name__ == "__main__":
    input_dir = Path("resources") /"transcripts" / "json"

    base_dir = Path("resources") / "transcripts" / "processed"
    sentence_dir = base_dir / "sentences"
    segment_dir = base_dir / "segments"

    base_dir.mkdir(parents=True, exist_ok=True)
    sentence_dir.mkdir(parents=True, exist_ok=True)
    segment_dir.mkdir(parents=True, exist_ok=True)
    input_paths = list(input_dir.glob("*.json"))[:15]
    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            fragments = json.load(f)

        sentences = merge_fragments_into_sentences(fragments)
        segments = segment_by_gap_and_keywords(sentences)

        base_name = input_path.stem  # filename without .json
        sentence_path = sentence_dir / f"{base_name}.json"
        segment_path = segment_dir / f"{base_name}.json"

        with open(sentence_path, "w", encoding="utf-8") as f:
            json.dump(sentences, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(sentences)} merged sentences to {sentence_path}")

        with open(segment_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(segments)} segments to {segment_path}")