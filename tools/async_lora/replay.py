"""Use the existing cross-environment OPD buffer with native Tinker losses."""

import asyncio
import math

from tinker import types

from async_lora.objective import opd_advantages
from slime.utils.opd_replay import replay_count, validate_trace


def select_replay(buffer, environment, step, seed, batch_size, fraction):
    if buffer is None:
        return []
    return buffer.sample(
        replay_count(step, batch_size, fraction), excluded_environments=[environment], seed=seed + step
    )


async def score_trajectories(rows, completions, sampler):
    if len(rows) != len(completions):
        raise ValueError("OPD requires one student completion per prompt")
    scores = await asyncio.gather(
        *(
            sampler.compute_logprobs_async(types.ModelInput.from_ints(row[0] + completion.tokens))
            for row, completion in zip(rows, completions, strict=True)
        )
    )
    traces = []
    for row, completion, logprobs in zip(rows, completions, scores, strict=True):
        prompt = row[0]
        if len(logprobs) != len(prompt) + len(completion.tokens):
            raise RuntimeError("Teacher scoring is misaligned with the student trajectory")
        teacher = logprobs[len(prompt) :]
        if any(v is None or not math.isfinite(v) for v in teacher):
            raise RuntimeError("Teacher returned a nonfinite response log probability")
        trace = {
            "tokens": prompt + completion.tokens,
            "response_length": len(completion.tokens),
            "rollout_log_probs": list(completion.logprobs),
            "teacher_log_probs": teacher,
            "loss_mask": list(completion.mask),
            "metadata": {"environment": row[1], "problem_difficulty": row[3]},
            "reward": {"reward": completion.reward, "accuracy": completion.accuracy},
            "status": "truncated" if completion.truncated else "completed",
            "opd_replay": False,
        }
        validate_trace(trace)
        traces.append(trace)
    return traces


def trace_datums(traces, batch_size):
    datums = []
    for trace in traces:
        validate_trace(trace)
        tokens, response_length = trace["tokens"], trace["response_length"]
        prefix = [0.0] * (len(tokens) - response_length - 1)
        advantages = opd_advantages(
            trace["rollout_log_probs"], trace["teacher_log_probs"], trace["loss_mask"], batch_size
        )
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tokens[1:],
                    "logprobs": prefix + trace["rollout_log_probs"],
                    "advantages": prefix + advantages,
                },
            )
        )
    return datums


def trajectory_metrics(traces):
    values = [-sum(opd_advantages(t["rollout_log_probs"], t["teacher_log_probs"], t["loss_mask"], 1)) for t in traces]
    return {"train/opd_reverse_kl": sum(values) / len(values)}


async def backward_batch(training, fresh, replayed, batch_size, clip_epsilon=0.2):
    """Accumulate two summed objectives, each normalized by the full mixed batch.

    One optimizer step follows both calls. Fresh samples retain unclipped OPD;
    cached, signed advantages use the existing PPO-clipped replay surrogate.
    """
    if len(fresh) + len(replayed) != batch_size or not fresh or not 0 < clip_epsilon < 1:
        raise ValueError("OPD mixed batch size or replay clipping is invalid")
    result = await (await training.forward_backward_async(trace_datums(fresh, batch_size), "importance_sampling"))
    loss = result.metrics.get("loss", 0)
    if replayed:
        result = await (
            await training.forward_backward_async(
                trace_datums(replayed, batch_size),
                "ppo",
                {"clip_low_threshold": 1 - clip_epsilon, "clip_high_threshold": 1 + clip_epsilon},
            )
        )
        loss += result.metrics.get("loss", 0)
    return {"train/opd_loss": loss}
