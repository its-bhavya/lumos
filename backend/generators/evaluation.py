import dspy

class AnswerEvaluationSignature(dspy.Signature):
    """
    Evaluate how well a user's answer matches a list of keypoints.

    Scoring rule (based only on keypoints coverage):
    - <50% of keypoints covered → score = 0
    - 50–80% covered → score = 0.5
    - >80% covered → score = 1

    """

    user_answer = dspy.InputField(desc="The student's answer in free-form text.")
    keypoints = dspy.InputField(desc="List of required key ideas the answer must include.")
    correct_answer = dspy.InputField(desc="Short model reference answer for context.")
    evaluation_json = dspy.OutputField(
        desc="""A raw JSON object which must contain
        - "score": 0|0.5|1 (based on scoring rule)
        - "coverage_percent": the percent of key points covered in the user_answer, integer 0-100
        - "missing_points": list of keypoints not mentioned
        - "evaluation_feedback": one-sentence feedback of user answer
        """
    )


class AnswerEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.Predict(AnswerEvaluationSignature)

    def forward(self, user_answer, keypoints, correct_answer):
        result = self.evaluate(
            user_answer=user_answer,
            keypoints=keypoints,
            correct_answer=correct_answer
        )
        return result
