"""Direct SGLang sampling with the established RLVE template and answer forcing."""

import asyncio
import hashlib

from composed_clustering.server import request


async def score_cell(
    url, adapter, environment, rows, tokenizer, settings, count, semaphore
):
    from lora_teachers.sampling import forcing_prefix, prompt_tokens

    from Gym.environments import identifier2environment

    async def score(row):
        prompt = row["user_prompt"]
        meta = row["metadata"]
        if isinstance(meta, str):
            import json

            meta = json.loads(meta)
        prompt_id = hashlib.sha256(prompt.encode()).hexdigest()
        seed = (1234 + int(prompt_id[:8], 16)) % (2**31)
        input_ids = prompt_tokens(tokenizer, prompt)

        async def completion(index):
            async with semaphore:
                params = {
                    "max_new_tokens": settings["eval-max-response-len"],
                    "temperature": settings["eval-temperature"],
                    "top_p": settings["eval-top-p"],
                    "sampling_seed": (seed + 1009 * index) % (2**31),
                }
                payload = {"input_ids": input_ids, "sampling_params": params}
                if adapter:
                    payload["lora_path"] = adapter
                response = await request(url + "/generate", payload)
                tokens = response["output_ids"]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                forced = (
                    response["meta_info"]["finish_reason"]["type"] == "length"
                    and settings["answer-forcing"]
                )
                if forced:
                    injected = tokenizer.encode(
                        forcing_prefix(text), add_special_tokens=False
                    )
                    continuation = {
                        **params,
                        "max_new_tokens": settings["answer-forcing-extra-tokens"],
                        "sampling_seed": (seed + 1009 * index + 104729) % (2**31),
                    }
                    tail = {
                        "input_ids": input_ids + tokens + injected,
                        "sampling_params": continuation,
                    }
                    if adapter:
                        tail["lora_path"] = adapter
                    result = await request(url + "/generate", tail)
                    text += forcing_prefix(text) + tokenizer.decode(
                        result["output_ids"], skip_special_tokens=True
                    )
                verifier = identifier2environment[environment](
                    answer_markers=(r"\boxed{", "}")
                )
                verifier.set_config(meta["config"])
                result = verifier.verifier(text)
                return int(result["accuracy"])

        values = await asyncio.gather(*(completion(i) for i in range(count)))
        return {"prompt_id": prompt_id, "correct": sum(values), "total": count}

    observations = await asyncio.gather(*(score(row) for row in rows))
    return {
        "environment": environment,
        "accuracy": sum(r["correct"] for r in observations)
        / sum(r["total"] for r in observations),
        "observations": observations,
    }
