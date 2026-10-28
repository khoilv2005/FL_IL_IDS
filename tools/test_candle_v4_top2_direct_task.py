import unittest
from types import SimpleNamespace
import numpy as np
from sklearn.preprocessing import StandardScaler
from debug_candle_v4_top2_direct_task import direct_task_scores, CACHE
import debug_candle_v4_top2_direct_task as implementation
from candle_v4_top2_task_select import choose_task


class DirectTaskTests(unittest.TestCase):
    def test_fit_only_saved_memory_and_query_does_not_change_fit(self):
        x = np.array([[-3., 0.], [-2., 0.], [2., 0.], [3., 0.]])
        memory = SimpleNamespace(memory={0:x[:2].copy(), 6:x[2:].copy()},
                                 class_to_task={0:0, 6:1}, scaler=StandardScaler().fit(x))
        scores = direct_task_scores(memory, x)
        np.testing.assert_array_equal(choose_task(np.array([[0,1]]*4), scores), [0,0,1,1])
        clf = CACHE[id(memory)][2]; before = clf.coef_.copy()
        direct_task_scores(memory, np.array([[1e6, -1e6]]))
        np.testing.assert_array_equal(clf.coef_, before)
        np.testing.assert_array_equal(memory.memory[0], x[:2])

    def test_single_task_and_unavailable_task(self):
        x = np.array([[1.,2.], [2.,3.]])
        memory = SimpleNamespace(memory={12:x},class_to_task={12:2},scaler=StandardScaler().fit(x))
        scores = direct_task_scores(memory,x)
        np.testing.assert_array_equal(scores[:,2], [1.,1.])
        np.testing.assert_array_equal(choose_task(np.array([[2,-1],[2,-1]]),scores),[2,2])

    def test_rbf_binary_decision_orientation(self):
        x = np.array([[-3.,0.],[-2.,0.],[2.,0.],[3.,0.]])
        memory = SimpleNamespace(memory={0:x[:2],6:x[2:]},class_to_task={0:0,6:1},scaler=StandardScaler().fit(x))
        implementation.NONLINEAR = True
        try:
            scores = direct_task_scores(memory,x)
            np.testing.assert_array_equal(choose_task(np.array([[0,1]]*4),scores),[0,0,1,1])
        finally:
            implementation.NONLINEAR = False


if __name__ == '__main__': unittest.main()
