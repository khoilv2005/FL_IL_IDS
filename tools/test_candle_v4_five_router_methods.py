import copy
import unittest
from types import SimpleNamespace
import numpy as np
from sklearn.preprocessing import StandardScaler
from pilot_candle_v4_five_router_methods import Scorers,METHODS
from candle_v4_top2_task_select import choose_task

class Tests(unittest.TestCase):
    def test_five_scores_separate_simple_memory_and_keep_missing_tasks_out(self):
        x=np.array([[-3.-i/10,-1.] for i in range(6)]+[[3.+i/10,1.] for i in range(6)])
        memory=SimpleNamespace(memory={0:x[:6].copy(),6:x[6:].copy()},scaler=StandardScaler().fit(x))
        before=copy.deepcopy(memory.memory);scorer=Scorers(memory)
        results=scorer.score(x)
        self.assertEqual(set(results),set(METHODS))
        for scores in results.values():
            self.assertTrue(np.isfinite(scores[:,:2]).all())
            self.assertTrue(np.isneginf(scores[:,2:]).all())
            np.testing.assert_array_equal(choose_task(np.array([[0,1]]*12),scores),[0]*6+[1]*6)
        for c in before:np.testing.assert_array_equal(before[c],memory.memory[c])

    def test_singleton_covariance_falls_back_to_shared_precision(self):
        x=np.array([[0.,0.],[2.,1.],[3.,2.]])
        memory=SimpleNamespace(memory={0:x[:1],6:x[1:]},scaler=StandardScaler().fit(x))
        scorer=Scorers(memory)
        np.testing.assert_array_equal(scorer.class_precision[0],scorer.precision)
        for scores in scorer.score(x).values():self.assertTrue(np.isfinite(scores[:,:2]).all())

if __name__=='__main__':unittest.main()
