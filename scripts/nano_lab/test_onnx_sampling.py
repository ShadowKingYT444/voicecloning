"""Small pure-NumPy checks for Nano's logit processor contract."""
import unittest
import numpy as np
from onnx_runtime import _sample, sampling_selfcheck


class CaptureChoice:
    def choice(self, count, p):
        self.probabilities=p
        return int(np.argmax(p))


class SamplingTests(unittest.TestCase):
    def test_top_p_boundary(self):
        self.assertEqual(sampling_selfcheck()["status"], "passed")

    def test_repetition_penalty_follows_top_p(self):
        rng=CaptureChoice()
        _sample(np.array([3.,2.,1.]),[0],temperature=1.,top_k=0,top_p=.7,repetition_penalty=5.,rng=rng)
        self.assertEqual(rng.probabilities[2],0.)
        self.assertGreater(rng.probabilities[0],0.)

    def test_zero_temperature_keeps_upstream_sampling_semantics(self):
        rng=CaptureChoice()
        _sample(np.array([3.,2.,1.]),[],temperature=0.,top_k=0,top_p=1.,repetition_penalty=1.,rng=rng)
        self.assertTrue(np.all(rng.probabilities>0))


if __name__=="__main__":
    unittest.main()
