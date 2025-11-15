import dspy


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
