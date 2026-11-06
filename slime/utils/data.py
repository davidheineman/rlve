import json
import random
import numpy as np
import pandas as pd
from datasets import Dataset as hf_ds

import base64
import zlib
import pickle

from slime.utils.types import Sample

__all__ = ["Dataset"]

from typing import Union, List, Dict


# TODO: don't read the whole file into memory.
def read_file(path):
    ds = None
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
    elif path.endswith(".json"): # Our custom stuff
        ds = hf_ds.from_json(path)
        df = None
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")
    
    if path.startswith("data/BENCHMARKS/LiveCodeBench") :
        assert path.endswith("jsonl")
        dataset = [row.to_dict() for _, row in df.iterrows()]
        df = None

        def translate_private_test_cases(encoded_data):
            decoded_data = base64.b64decode(encoded_data)
            decompressed_data = zlib.decompress(decoded_data)
            original_data = pickle.loads(decompressed_data)
            return json.loads(original_data)

        def has_test_type(tests, type):  ## helper to select specific type of problems
            """
            Check if any test in the test list has 'testtype' set to 'type'.
            """
            test_list = json.loads(tests)
            for test in test_list:
                if test.get("testtype") == type:
                    return True
            return False

        def map_to_example(row):
            return {
                "prompt": row["question_content"],
                "test": row["private_test_cases"],
                "entry_point": row["starter_code"],
                "task_id": row["question_id"],
                "is_stdin": has_test_type(row["public_test_cases"], "stdin"),
                "public_test_cases": row["public_test_cases"],
                "difficulty": row["difficulty"],
            }

        ds = []
        for instance in dataset :
            instance["private_test_cases"] = translate_private_test_cases(instance["private_test_cases"])
            instance = map_to_example(instance)

            generic_prompt = "You will be given a question (problem specification) and must generate a correct Python program that meets the specification and passes all tests."
            if instance["is_stdin"]:
                prompt_text = generic_prompt + "\n" + "Ensure that the Python program runs correctly when executed directly, without requiring any extra code or function calls; it reads the input from stdin, solves the problem, and writes the output to stdout (do not test directly on the sample inputs)." + "\n\n" + instance["prompt"]
            else:
                prompt_text = generic_prompt + "\n" + "Generate an executable Python function based on the given question." + "\n\n" + instance["prompt"] + "\n\n" + "You will use the following starter code in the final solution (please return only the function body, without invoking it in the final solution):\n" + f"```python\n" + instance["entry_point"] + f"\n```"

            ds.append(dict(
                user_prompt = prompt_text,
                metadata = dict(
                    is_stdin = has_test_type(instance["public_test_cases"], "stdin"),
                    test = instance["test"],
                ),
            ))
    
    if df is None :
        assert ds is not None
        for data in ds:
            yield data
        return
    for _, row in df.iterrows():
        yield row.to_dict()


TinyZero_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Show your work in <think> </think> tags, and return the final answer in <answer> </answer> tags.
User: {prompt}
Assistant: Let me solve this step by step.
<think>"""
# RLVE environment prompts never say how to mark the final answer; the verifier just looks
# for the answer markers. Reasoning models (R1-Distill, ProRL, OpenThinker) emit \boxed{}
# on their own, but a plain instruct model answers "21" and scores -1 on every sample, which
# leaves GRPO with no reward variance at all. This suffix asks for the marker explicitly.
BOXED_ANSWER_SUFFIX = "\n\nThink step by step, then give your final answer inside \\boxed{}."
def custom_prompt_preprocessor(args, user_prompt : str, apply_chat_template : bool) -> Union[str, List[Dict[str, str]]] :
    if args.custom_prompt_preprocessor == "TinyZero" :
        assert not apply_chat_template
        return TinyZero_TEMPLATE.format(prompt = user_prompt)
    elif args.custom_prompt_preprocessor in ("ChatTemplate_NoSystemPrompt", "ChatTemplate_BoxedAnswer") :
        assert apply_chat_template
        # Check if user_prompt is already in chat format (list of dicts)
        if isinstance(user_prompt, list) and len(user_prompt) > 0 and isinstance(user_prompt[0], dict) and "role" in user_prompt[0]:
            return user_prompt
        if args.custom_prompt_preprocessor == "ChatTemplate_BoxedAnswer" :
            assert args.answer_marker_type == r"\boxed{}", "ChatTemplate_BoxedAnswer only makes sense with --answer-marker-type '\boxed{}'"
            user_prompt = user_prompt + BOXED_ANSWER_SUFFIX
        return [{"role" : "user", "content": user_prompt}]
    else :
        raise NotImplementedError(f"User prompt processor {args.custom_prompt_preprocessor} not implemented.")


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        max_length,
        *,
        prompt_key="text",
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        args=None,
    ):
        self.args = args

        self.origin_samples = []
        for data in read_file(path):
            prompt = data[prompt_key]
            prompt = custom_prompt_preprocessor(args = self.args, user_prompt = prompt, apply_chat_template = apply_chat_template)
            if apply_chat_template:
                if tool_key is not None:
                    tools = data[tool_key]
                    if isinstance(tools, str):
                        tools = json.loads(tools)
                    elif isinstance(tools, np.ndarray):
                        tools = tools.tolist()
                    assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                else:
                    tools = None
                prompt = tokenizer.apply_chat_template(prompt, tools, tokenize=False, add_generation_prompt=True)

            # TODO: this is slow.
            if max_length is not None:
                # assert False, "For now, we don't discard overlong prompts"
                if len(tokenizer(prompt)["input_ids"]) > max_length:
                    continue
            
            metadata = data.get(metadata_key) or {}
            if hasattr(self.args, "eval_max_context_len") and self.args.eval_max_context_len is not None :
                metadata["input_length"] = len(tokenizer(prompt)["input_ids"])
            self.origin_samples.append(
                Sample(
                    prompt=prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                )
            )

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
