import hashlib
import copy
import json
import random
from pathlib import Path


def load_candidate(path, index, environments, pool_path):
    manifest = json.loads(Path(path).read_text())
    if manifest["environments"] != list(environments):
        raise ValueError("mixture environments do not match the training environment list")
    candidates = manifest["candidates"]
    if index < 0 or index >= len(candidates):
        raise ValueError(f"mixture index {index} is outside 0..{len(candidates) - 1}")
    actual_hash = hashlib.sha256(Path(pool_path).read_bytes()).hexdigest()
    if actual_hash != manifest["pool_sha256"]:
        raise ValueError("frozen prompt pool differs from the mixture manifest")
    candidate = candidates[index]
    weights = candidate["weights"]
    if set(weights) != set(environments) or any(weight <= 0 for weight in weights.values()):
        raise ValueError("mixture weights must be positive for every training environment")
    if abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("mixture weights must sum to one")
    return manifest, candidate


class FrozenPromptMix:
    def __init__(self, path, environments, weights, seed, expected_sha256):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("frozen prompt pool changed after mixture selection")
        self.environments = list(environments)
        self.weights = [weights[environment] for environment in environments]
        self.by_environment = {environment: [] for environment in environments}
        for record in json.loads(raw):
            metadata = record["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            environment = metadata["environment"]
            if environment not in self.by_environment:
                raise ValueError(f"unexpected prompt-pool environment {environment}")
            self.by_environment[environment].append((record["user_prompt"], metadata["config"]))
        if any(not records for records in self.by_environment.values()):
            raise ValueError("every training environment needs prompts in the frozen pool")
        self.environment_rng = random.Random(seed)
        self.prompt_rng = random.Random(seed + 1)

    def draw(self):
        environment = self.environment_rng.choices(self.environments, self.weights, k=1)[0]
        prompt, config = self.prompt_rng.choice(self.by_environment[environment])
        return environment, prompt, copy.deepcopy(config)

    def get_state(self):
        return self.environment_rng.getstate(), self.prompt_rng.getstate()

    def set_state(self, state):
        environment_state, prompt_state = state
        self.environment_rng.setstate(environment_state)
        self.prompt_rng.setstate(prompt_state)
