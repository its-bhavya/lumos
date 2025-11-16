import dspy, re, json

def clean_json_field(field:str):
            field = re.sub(r"^```(?:json)?\n?", "", field.strip())
            field = re.sub(r"\n?```$", "", field)
            return json.loads(field)

class StructureExtractionSignature(dspy.Signature):

    transcript = dspy.InputField(desc="The full transcript of a lecture or explanation.")
    central_topic = dspy.OutputField(desc="The central concept of the transcript.")
    subtopics = dspy.OutputField(
        desc="""A raw JSON list where each subtopic has:
        - 'title': string
        - 'description': a brief phrase 
        - optional 'children': a list of similar subtopic objects
        If more explanation is needed, use nested subtopics under 'children' instead of a long description.
        Subtopics can have their own subtopics up till it is relevant."""
    )

class MindmapExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.teleprompt = dspy.ChainOfThought(StructureExtractionSignature)

    def forward(self, transcript: str):
        return self.teleprompt(transcript=transcript)

class KeyPointExtractorSignature(dspy.Signature):
    answer = dspy.InputField(desc="The correct answer for a quiz question.")
    key_points = dspy.OutputField(
        desc=(
            """
            A JSON list of the essential words/phrases explicitly present in the answer.   
            - Do NOT infer or add extra definitions, properties, or facts.
            -  If the answer is a single phrase (e.g., "BCNF"), return a list with exactly that phrase.        
            """
        )
    )

class KeyPointExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(KeyPointExtractorSignature)

    def forward(self, answer: str):
        return self.generator(answer=answer)
    
key_point_extractor = KeyPointExtractor()

def extract_key_points(answer: str):
    result = key_point_extractor(answer=answer)
    raw = result.key_points
    return clean_json_field(raw)

def evaluate_answer(correct_answer, user_answer):
    expected_points = extract_key_points(correct_answer)

    matched = 0
    missed = []

    user_lower = user_answer.lower()

    for p in expected_points:
        if p.lower() in user_lower:
            matched += 1
        else:
            missed.append(p)

    coverage = matched / len(expected_points)

    if coverage < 0.5:
        score = 0
        status = "incorrect"
    elif coverage < 0.8:
        score = 0.5
        status = "good but could be better"
    else:
        score = 1
        status = "great"

    return {
        "score": score,
        "coverage_percent": round(coverage * 100, 2),
        "missed_points": missed,
        "status": status
    }

