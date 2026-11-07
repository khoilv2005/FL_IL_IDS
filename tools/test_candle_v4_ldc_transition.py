import unittest
import numpy as np
import torch
from pilot_candle_v4_ldc_transition import Compensation

class Tests(unittest.TestCase):
    def test_ridge_recovers_translation_on_unseen_pairs(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(42);x=rng.normal(size=(100,3)).astype(np.float32)
        y=(2*x+np.array([1.,-2.,4.])).astype(np.float32)
        mapper=Compensation(x[:80],y[:80]);before=mapper.ridge.copy()
        prediction=mapper.transform(x[80:],'ridge')
        self.assertLess(np.mean((prediction-y[80:])**2),.01)
        np.testing.assert_array_equal(before,mapper.ridge)
        self.assertTrue(np.isfinite(mapper.transform(x[80:],'mlp')).all())

if __name__=='__main__':unittest.main()
