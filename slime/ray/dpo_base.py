"""A frozen base-policy sampler on the fifth GPU of an online DPO run."""

import os

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer

from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from slime.utils.types import Sample


@ray.remote(num_cpus=1, num_gpus=1)
class BaseGenerator:
    def __init__(self, model_path, max_new_tokens):
        visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        assigned = str(ray.get_gpu_ids()[0])
        if assigned not in visible:
            raise RuntimeError(f"Base GPU {assigned} is not in {visible}")
        self.device = torch.device(f"cuda:{visible.index(assigned)}")
        torch.cuda.set_device(self.device)
        torch.set_num_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval().to(self.device)
        self.model.requires_grad_(False)
        self.max_new_tokens = max_new_tokens

    def ready(self):
        return True

    @torch.inference_mode()
    def generate(self, sample_dicts):
        samples = [Sample.from_dict(row) for row in sample_dicts]
        prompts = [sample.prompt for sample in samples]
        encoded = self.tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model.generate(
            **inputs, do_sample=True, temperature=1.0, top_p=1.0,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        width = inputs["input_ids"].shape[1]
        eos = self.tokenizer.eos_token_id
        for sample, output in zip(samples, outputs):
            response_ids = output[width:].tolist()
            finished = eos in response_ids
            if finished:
                response_ids = response_ids[:response_ids.index(eos) + 1]
            prompt_ids = self.tokenizer.encode(sample.prompt, add_special_tokens=False)
            sample.tokens = prompt_ids + response_ids
            sample.response_length = len(response_ids)
            sample.response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            sample.status = Sample.Status.COMPLETED if finished else Sample.Status.TRUNCATED
        return [sample.to_dict() for sample in samples]


def create_base_generator(args, placement):
    placement_group, bundle_indices = placement
    if len(bundle_indices) != 1:
        raise RuntimeError("Online DPO needs one base generator GPU")
    generator = BaseGenerator.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=bundle_indices[0],
        ),
        runtime_env={"env_vars": {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}},
    ).remote(args.dpo_base_hf, args.rollout_max_response_len)
    ray.get(generator.ready.remote())
    return generator
