"""An interrupted generation must never become a partial training trajectory."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import tinker
from async_lora.flex_sampling import ResilientSampler, ResilientSamplingService


def abort(message="the engine aborted this sample; resubmit the request"):
    return tinker.RequestFailedError(message, request_id="request", category="server")


class FlexSamplingTest(unittest.TestCase):
    def test_aborted_samples_and_scoring_retry_exact_arguments_on_same_client(self):
        for method in ("sample_async", "compute_logprobs_async"):
            original = AsyncMock(side_effect=[abort(), "complete"])
            sampler = ResilientSampler(SimpleNamespace(**{method: original}, model_path="immutable-v7"))
            kwargs = {"prompt": [1, 2], "sampling_params": {"seed": 1234}, "num_samples": 8}
            with patch("async_lora.flex_sampling.asyncio.sleep", new=AsyncMock()):
                self.assertEqual(asyncio.run(getattr(sampler, method)(**kwargs)), "complete")
            self.assertEqual(original.await_count, 2)
            self.assertTrue(all(call.kwargs == kwargs for call in original.await_args_list))
            self.assertEqual(sampler.model_path, "immutable-v7")

    def test_persistent_aborts_are_bounded_and_other_errors_fail_immediately(self):
        for error, expected in ((abort(), 6), (abort("wrong adapter configuration"), 1), (ValueError("bad data"), 1)):
            original = AsyncMock(side_effect=error)
            sampler = ResilientSampler(SimpleNamespace(sample_async=original))
            with patch("async_lora.flex_sampling.asyncio.sleep", new=AsyncMock()), self.assertRaises(type(error)):
                asyncio.run(sampler.sample_async(prompt=[1]))
            self.assertEqual(original.await_count, expected)

    def test_service_wraps_sampling_only_and_passes_immutable_model_reference(self):
        original = SimpleNamespace(
            create_sampling_client_async=AsyncMock(return_value=SimpleNamespace(model_path="immutable-v7")),
            create_lora_training_client_async=object(),
        )
        service = ResilientSamplingService(original)
        sampler = asyncio.run(service.create_sampling_client_async(model_path="immutable-v7"))
        self.assertIsInstance(sampler, ResilientSampler)
        original.create_sampling_client_async.assert_awaited_once_with(model_path="immutable-v7")
        self.assertIs(service.create_lora_training_client_async, original.create_lora_training_client_async)


if __name__ == "__main__":
    unittest.main()
