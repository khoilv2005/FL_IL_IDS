import copy
import unittest
from types import SimpleNamespace
import numpy as np
from sklearn.preprocessing import StandardScaler
from pilot_candle_v4_ovr_task_router import OVRScorers,METHODS
from candle_v4_top2_task_select import choose_task

class Tests(unittest.TestCase):
    def test_task_and_class_negative_discrimination_without_query_fit(self):
        x=np.array([[-3.,-1.],[-2.,-1.],[2.,1.],[3.,1.]])
        m=SimpleNamespace(memory={0:x[:2].copy(),6:x[2:].copy()},scaler=StandardScaler().fit(x))
        before=copy.deepcopy(m.memory);scorers=OVRScorers(m)
        scores=scorers.score(x);self.assertEqual(set(scores),set(METHODS))
        for s in scores.values():
            np.testing.assert_array_equal(choose_task(np.array([[0,1]]*4),s),[0,0,1,1])
            self.assertTrue(np.isneginf(s[:,2:]).all())
        for c in before:np.testing.assert_array_equal(before[c],m.memory[c])
    def test_single_task_and_single_class(self):
        x=np.array([[1.,2.],[2.,3.]])
        m=SimpleNamespace(memory={12:x},scaler=StandardScaler().fit(x))
        for s in OVRScorers(m).score(x).values():
            np.testing.assert_array_equal(choose_task(np.array([[2,-1]]*2),s),[2,2])

if __name__=='__main__':unittest.main()
