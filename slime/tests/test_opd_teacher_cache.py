import importlib.util
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location("opd_teacher_cache", Path(__file__).parents[1] / "utils/opd_teacher_cache.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TeacherModelCacheTest(unittest.TestCase):
    def test_eviction_precedes_load_and_hits_do_not_transfer_weights(self):
        resident, transfers = set(), []

        class Model:
            def __init__(self, name):
                self.name = name

            def to(self, device):
                transfers.append((self.name, device))
                if device == "cpu":
                    resident.remove(self.name)
                else:
                    resident.add(self.name)
                self_test.assertLessEqual(len(resident), 2)
                return self

        self_test = self
        models = {name: Model(name) for name in ("A", "B", "C")}
        cache = module.TeacherModelCache(models, "cuda:0", 2)
        cache.get("A")
        cache.get("B")
        cache.get("A")
        cache.get("C")
        self.assertEqual(transfers, [("A", "cuda:0"), ("B", "cuda:0"), ("B", "cpu"), ("C", "cuda:0")])
        self.assertIs(cache.get("A"), models["A"])
        self.assertEqual(set(cache.resident), {"A", "C"})
        cache.get("B")
        self.assertEqual(set(cache.resident), {"A", "B"})

    def test_invalid_capacity_or_unknown_environment_does_not_evict(self):
        with self.assertRaises(ValueError):
            module.TeacherModelCache({}, "cuda:0", 0)
        cache = module.TeacherModelCache({}, "cuda:0", 2)
        with self.assertRaises(KeyError):
            cache.get("missing")
        self.assertEqual(len(cache.resident), 0)
