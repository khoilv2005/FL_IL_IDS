"""Task selection policy contracts; no real data or training."""
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from candle_v4_top2_task_select import task_mass,choose_task,predict_selected,FixedChoiceView,task_prototype_similarity


class SelectTests(unittest.TestCase):
    def test_sum_class_mass_can_change_task_without_unmasking_both(self):
        mem=SimpleNamespace(fitted_classes=np.array([0,6,7]),scaler=SimpleNamespace(transform=lambda x:x),
            classifier=SimpleNamespace(classes_=np.array([0,6,7]),predict_proba=lambda x:np.array([[.4,.3,.3]])))
        mass=task_mass(mem,np.zeros((1,2)));np.testing.assert_allclose(mass[0,:2],[.4,.6])
        np.testing.assert_array_equal(choose_task(np.array([[0,1]]),mass),[1])

    def test_tie_or_absent_second_preserves_top1(self):
        m=np.zeros((3,6));m[0,:2]=.5;m[1,0]=1;m[2,2]=.1;m[2,3]=.9
        np.testing.assert_array_equal(choose_task(np.array([[0,1],[0,-1],[3,2]]),m),[0,0,3])

    def test_cannot_select_third_task_even_if_it_has_more_mass(self):
        m=np.array([[.2,.3,.5,0,0,0]])
        np.testing.assert_array_equal(choose_task(np.array([[0,1]]),m),[1])

    def test_single_class_memory_stays_single_task(self):
        m=task_mass(SimpleNamespace(fitted_classes=np.array([12])),np.zeros((4,3)))
        np.testing.assert_array_equal(m[:,2],np.ones(4))
        np.testing.assert_array_equal(choose_task(np.array([[2,-1]]*4),m),np.full(4,2))

    def test_invalid_probabilities_rejected(self):
        mem=SimpleNamespace(fitted_classes=np.array([0,6]),scaler=SimpleNamespace(transform=lambda x:x),
            classifier=SimpleNamespace(classes_=np.array([0,6]),predict_proba=lambda x:np.array([[np.nan,.5]])))
        with self.assertRaises(ValueError):task_mass(mem,np.zeros((1,2)))

    def test_predictor_no_labels_and_fixed_adapter_route(self):
        self.assertEqual(list(inspect.signature(predict_selected).parameters),['service','cid','model','det','inputs','seen','rule'])
        view=FixedChoiceView(None,None,[1,0])
        np.testing.assert_array_equal(view.predict_episodes_batch(np.zeros((2,3))),[1,0])
        with self.assertRaises(ValueError):view.predict_episodes_batch(np.zeros((1,3)))

    def test_receiver_prototypes_stay_in_the_same_standardized_coordinates(self):
        mem=SimpleNamespace(memory={0:np.array([[1.,0.],[1.,0.]]),6:np.array([[0.,1.],[0.,1.]])},scaler=SimpleNamespace(transform=lambda x:x))
        scores=task_prototype_similarity(mem,np.array([[1.,0.],[0.,1.]]))
        np.testing.assert_array_equal(choose_task(np.array([[0,1],[0,1]]),scores),[0,1])


if __name__=='__main__':unittest.main()
