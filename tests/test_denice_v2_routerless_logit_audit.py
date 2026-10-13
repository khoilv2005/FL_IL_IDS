"""Pure inference/statistics checks; no optimizer or data fitting."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from tools.audit_denice_v2_routerless_old_logits import masked_argmax,margin_summary,ranking_auc,write_ecdf_svg


class LogitAuditTests(unittest.TestCase):
    def test_old_only_does_not_let_new_logits_win(self):
        logits=np.array([[1.,2.,100.],[3.,1.,200.]])
        np.testing.assert_array_equal(masked_argmax(logits,(0,1)),[1,0])
        np.testing.assert_array_equal(masked_argmax(logits,(0,1,2)),[2,2])

    def test_noncontiguous_real_class_mapping(self):
        logits=np.zeros((1,12));logits[0,7]=3;logits[0,10]=5
        np.testing.assert_array_equal(masked_argmax(logits,(6,7,8,10)),[10])

    def test_invalid_masks_rejected(self):
        for classes in ((),(0,0),(-1,),(8,)):
            with self.assertRaises(ValueError):
                masked_argmax(np.zeros((1,3)),classes)

    def test_selected_nonfinite_logits_rejected(self):
        with self.assertRaises(ValueError):
            masked_argmax(np.array([[np.nan,1.]]),(0,1))

    def test_auc_direction_and_ties(self):
        self.assertEqual(ranking_auc(np.array([1.,2.]),np.array([3.,4.])),1)
        self.assertEqual(ranking_auc(np.array([3.,4.]),np.array([1.,2.])),0)
        self.assertEqual(ranking_auc(np.array([1.]),np.array([1.])),.5)

    def test_distribution_all_rows_and_zero_boundary(self):
        summary=margin_summary(np.array([-1.,0.,1.]))
        self.assertEqual(summary['rows'],3)
        self.assertEqual(summary['quantiles']['0.5'],0)
        self.assertEqual(summary['positive_fraction'],1/3)
        self.assertEqual(summary['zero_fraction'],1/3)

    def test_plot_writes_both_formats(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'margin.svg'
            write_ecdf_svg(path,np.array([.1,.2]),np.array([.3,.4]))
            self.assertIn('<svg',path.read_text(encoding='utf-8'))
            self.assertGreater(path.with_suffix('.png').stat().st_size,1000)


if __name__=='__main__':
    unittest.main()
