"""Retry aborted inference against the same immutable policy after worker loss."""

import asyncio

import tinker


async def retry_aborted(method, *args, **kwargs):
    for attempt in range(6):
        try:
            return await method(*args, **kwargs)
        except tinker.RequestFailedError as error:
            if "the engine aborted this sample; resubmit the request" not in str(error) or attempt == 5:
                raise
            print(f"FLEX_GENERATOR_ABORT_RETRY attempt={attempt + 1}", flush=True)
            await asyncio.sleep(1)


class ResilientSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    async def sample_async(self, *args, **kwargs):
        return await retry_aborted(self.sampler.sample_async, *args, **kwargs)

    async def compute_logprobs_async(self, *args, **kwargs):
        return await retry_aborted(self.sampler.compute_logprobs_async, *args, **kwargs)


class ResilientSamplingService:
    def __init__(self, service):
        self.service = service

    def __getattr__(self, name):
        return getattr(self.service, name)

    async def create_sampling_client_async(self, *args, **kwargs):
        return ResilientSampler(await self.service.create_sampling_client_async(*args, **kwargs))
