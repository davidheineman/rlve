"""Bound the GPU footprint of an environment-specific teacher bank."""

from collections import OrderedDict


class TeacherModelCache:
    def __init__(self, models, device, capacity):
        if capacity < 1:
            raise ValueError("Teacher GPU cache capacity must be positive")
        self.models = models
        self.device = device
        self.capacity = capacity
        self.resident = OrderedDict()

    def get(self, environment):
        if environment in self.resident:
            self.resident.move_to_end(environment)
            return self.models[environment]
        model = self.models[environment]
        if len(self.resident) == self.capacity:
            evicted, _ = self.resident.popitem(last=False)
            self.models[evicted].to("cpu")
        model.to(self.device)
        self.resident[environment] = None
        return model
