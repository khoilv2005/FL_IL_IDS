import unittest
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from methods import recursive_ridge,ComponentScorers,METHODS

class Tests(unittest.TestCase):
    def test_recursive_solution_matches_joint_ridge(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(30,5));tasks=np.repeat([0,1,2],10)
        y=rng.normal(size=(30,3));actual=recursive_ridge(x,y,tasks)
        np.testing.assert_allclose(actual,np.linalg.solve(x.T@x+np.eye(5),x.T@y),atol=1e-10)
    def test_six_methods_finite_singleton_and_memory_unchanged(self):
        torch.set_num_threads(2)
        x=np.array([[-2.,0.],[-3.,0.],[3.,1.]])
        mem=SimpleNamespace(memory={0:x[:2].copy(),6:x[2:].copy()},scaler=StandardScaler().fit(x))
        scores=ComponentScorers(mem).score(x)
        self.assertEqual(set(scores),set(METHODS))
        for s in scores.values():
            self.assertTrue(np.isfinite(s[:,:2]).all());self.assertTrue(np.isneginf(s[:,2:]).all())
        np.testing.assert_array_equal(mem.memory[0],x[:2])

if __name__=='__main__':unittest.main()
