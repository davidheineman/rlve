"""GPU-resident environment-specific teacher scorers for OPD."""

import json
import os
from pathlib import Path

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from slime.utils.opd import score_response_log_probs
from slime.utils.opd_teacher_marker import validate_teacher_marker
from slime.utils.opd_teacher_cache import TeacherModelCache


@ray.remote(num_cpus=1, num_gpus=1)
class TeacherScorer:
    def __init__(self, teacher_paths, cache_size=0, stop_token_permutation=None):
        self.stop_token_permutation = stop_token_permutation
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        assigned_device = str(ray.get_gpu_ids()[0])
        if assigned_device not in visible_devices:
            raise RuntimeError(f"Teacher GPU {assigned_device} is not in {visible_devices}")
        self.device = torch.device(f"cuda:{visible_devices.index(assigned_device)}")
        torch.cuda.set_device(self.device)
        torch.set_num_threads(1)
        self.models = {}
        for environment, directory in teacher_paths:
            marker = json.loads((Path(directory).parent / "weights_only_complete.json").read_text())
            validate_teacher_marker(marker, environment, directory)
            model = AutoModelForCausalLM.from_pretrained(
                directory,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            ).eval()
            if cache_size == 0:
                model.to(self.device)
            model.requires_grad_(False)
            self.models[environment] = model
        self.cache = TeacherModelCache(self.models, self.device, cache_size) if cache_size else None
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        print(
            f"Teacher GPU {assigned_device}: {len(self.models)} models, cache={cache_size or 'all'}, "
            f"{free_bytes / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB free",
            flush=True,
        )

    def ready(self):
        return sorted(self.models)

    def score(self, trajectories):
        results = []
        by_environment = {}
        for trajectory in trajectories:
            by_environment.setdefault(trajectory[1], []).append(trajectory)
        for environment, group in by_environment.items():
            model = self.cache.get(environment) if self.cache else self.models[environment]
            for index, _, tokens, response_length in group:
                scores = score_response_log_probs(model, tokens, response_length, self.device,
                    stop_token_permutation=self.stop_token_permutation)
                results.append((index, scores))
        return results


def create_teacher_scorers(args, placement):
    placement_group, bundle_indices = placement
    if len(bundle_indices) != args.opd_teacher_gpus:
        raise RuntimeError("Teacher GPU placement does not match configuration")
    scorer_actors = []
    for scorer_index, bundle_index in enumerate(bundle_indices):
        teacher_paths = [
            (
                environment,
                str(Path(args.opd_teacher_dir) / f"{index:02d}-{environment}" / "hf-final"),
            )
            for index, environment in enumerate(args.environment_list)
            if index % args.opd_teacher_gpus == scorer_index
        ]
        scorer = TeacherScorer.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_index,
            ),
            runtime_env={"env_vars": {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}},
        ).remote(teacher_paths, getattr(args, "opd_teacher_cache_size", 0),
                 ((args.opd_student_eos_token_id, args.opd_teacher_eos_token_id)
                  if getattr(args, "opd_student_eos_token_id", None) is not None else None))
        scorer_actors.append(scorer)
    initialized = ray.get([scorer.ready.remote() for scorer in scorer_actors])
    if sorted(sum(initialized, [])) != sorted(args.environment_list):
        raise RuntimeError("Teacher scorers did not load every training environment")
    return scorer_actors
