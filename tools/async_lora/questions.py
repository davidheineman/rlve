"""One immutable, scoreable training question per environment."""

import hashlib
import json
from copy import deepcopy


def question_id(row):
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    payload = {
        "user_prompt": row["user_prompt"],
        "environment": metadata["environment"],
        "config": metadata["config"],
        "difficulty": metadata["difficulty"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_questions(path, environments):
    data = json.loads(path.read_text())
    if data.get("version") != 1 or data["environments"] != list(environments):
        raise ValueError("Fixed-question data does not match the experiment panel")
    result = {}
    for row in data["questions"]:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if metadata["question_id"] != question_id(row):
            raise ValueError("Fixed training question changed")
        environment = metadata["environment"]
        if metadata["difficulty"] != data["difficulty"]:
            raise ValueError("Fixed-question difficulty differs from its manifest")
        if environment in result or environment not in environments:
            raise ValueError("Expected exactly one fixed question per environment")
        result[environment] = deepcopy(row)
        result[environment]["metadata"] = deepcopy(metadata)
    if list(result) != list(environments):
        raise ValueError("Fixed-question data is incomplete or out of order")
    return result


def evaluation_rows(questions, environment=None, count=32):
    selected = [questions[environment]] if environment is not None else list(questions.values())
    if count < len(selected) or count % len(selected):
        raise ValueError("Same-question evaluation must weight environments equally")
    return [deepcopy(row) for _ in range(count // len(selected)) for row in selected]


class FrozenProblem:
    def __init__(self, row):
        self.row = deepcopy(row)

    def prompt_generator(self):
        return self.row["user_prompt"]

    def get_config(self):
        metadata = self.row["metadata"]
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        return deepcopy(metadata["config"])


class SingleQuestionSource:
    def __init__(self, row):
        self.problem = FrozenProblem(row)
        self.metadata = row["metadata"]
        if isinstance(self.metadata, str):
            self.metadata = json.loads(self.metadata)
        self.draws = 0

    def generate_problem(self):
        self.draws += 1
        return self.metadata["environment"], self.metadata["difficulty"], self.problem

    def get_state(self):
        return {"question_id": self.metadata["question_id"], "draws": self.draws}

    def set_state(self, state):
        if (
            state["question_id"] != self.metadata["question_id"]
            or type(state["draws"]) is not int
            or state["draws"] < 0
        ):
            raise ValueError("Fixed-question source cannot resume with a different question")
        self.draws = state["draws"]
