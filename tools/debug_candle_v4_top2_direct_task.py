"""Bounded development audit: task-supervised reranker fitted on saved TRAIN features.

One fixed rule, no TEST fit or parameter search. Reuses the locked prior panel.
Does not change production inference or checkpoint state.
"""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import debug_candle_v4_top2_task_select as audit
import candle_v4_top2_task_select as selector

CACHE = {}
NONLINEAR = False


def direct_task_scores(memory, features):
    key = id(memory)
    if key not in CACHE:
        classes = sorted(memory.memory)
        x = np.concatenate([memory.memory[c] for c in classes])
        y = np.concatenate([np.full(len(memory.memory[c]), memory.class_to_task[c]) for c in classes])
        tasks = np.unique(y)
        clf = None
        if len(tasks) > 1:
            clf = (SVC(C=1., kernel='rbf', gamma='scale', class_weight='balanced',
                       decision_function_shape='ovr') if NONLINEAR else
                   LogisticRegression(C=1., max_iter=1000, solver='lbfgs',
                                      class_weight='balanced', random_state=0))
            clf.fit(memory.scaler.transform(x), y)
        CACHE[key] = (memory, tasks, clf)
    _, tasks, clf = CACHE[key]
    scores = np.zeros((len(features), 6))
    if clf is None:
        scores[:, tasks] = 1.
    elif isinstance(clf, SVC):
        decision = clf.decision_function(memory.scaler.transform(features))
        if decision.ndim == 1: decision = np.column_stack([-decision, decision])
        scores[:, tasks] = decision
    else:
        scores[:, tasks] = clf.predict_proba(memory.scaler.transform(features))
    return scores


if __name__ == '__main__':
    NONLINEAR = '--rbf' in sys.argv
    # Substitute only the audit's scoring callback, never production model state.
    selector.task_mass = direct_task_scores
    audit.POLICY = 'top2_saved_train_direct_task_rbf_v1' if NONLINEAR else 'top2_saved_train_direct_task_lr_v1'
    sys.argv = [sys.argv[0], '--rule', 'mass']
    # Redirect the old audit output without overwriting its receipt.
    original_root = audit.ROOT
    original_write = Path.write_text
    def write_receipt(path, data, *args, **kwargs):
        if path == original_root / 'audit_denice/candle_v4_top2_task_select_20261029/audit.json':
            name = 'candle_v4_top2_direct_task_rbf_20261029' if NONLINEAR else 'candle_v4_top2_direct_task_20261029'
            path = original_root / 'audit_denice' / name / 'audit.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            obj = json.loads(data)
            obj.update(no_refit=False, no_training=False, backbone_training=False,
                       reranker_fit_source='Existing saved TRAIN activation memory only',
                       scientific_status='Repeated TEST development panel; not independent confirmation; fixed C=1 task classifier',
                       classifier='SVC RBF gamma=scale, balanced, decision scores' if NONLINEAR else 'LR balanced')
            data = json.dumps(obj, indent=2)
        return original_write(path, data, *args, **kwargs)
    Path.write_text = write_receipt
    try:
        audit.main()
    finally:
        Path.write_text = original_write
