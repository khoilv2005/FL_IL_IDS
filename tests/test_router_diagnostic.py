import ast
from pathlib import Path
import unittest
import numpy as np


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        tree = ast.parse((self.root/'tools/diagnose_cluster_servers.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='masked_predictions')
        ns = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<test>', 'exec'), ns)
        self.predict = ns['masked_predictions']

    def test_oracle_changes_task_mask_not_logits(self):
        logits = np.array([[9., 1., 4., 8.], [3., 7., 8., 2.]])
        registered = {0: [0, 1], 1: [2, 3]}
        np.testing.assert_array_equal(self.predict(logits, np.array([0, 0]), registered), [0, 1])
        np.testing.assert_array_equal(self.predict(logits, np.array([1, 1]), registered), [3, 2])

    def test_missing_task_is_failure_not_dropped(self):
        np.testing.assert_array_equal(self.predict(np.ones((2, 4)), np.array([1, 2]), {1: []}), [-1, -1])

    def test_single_file_worker_matches_source_and_compiles(self):
        ns = {}
        tree = ast.parse((self.root/'eval_selective.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith('DIAGNOSTIC_') for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<constants>', 'exec'), ns)
        self.assertTrue(ns['DIAGNOSTIC_SOURCE'].endswith((self.root/'tools/diagnose_cluster_servers.py').read_text()))
        compile(ns['DIAGNOSTIC_SOURCE']+ns['DIAGNOSTIC_ENTRY'], '<worker>', 'exec')


if __name__ == '__main__':
    unittest.main()
