# Validation — 2026-10-28

## Scope and measured communication

Synthetic CoverageData fixture: CPU, two tasks, two rounds/task, unequal class
coverage across clusters. No TEST read during training. These are application
message accounting results from a simulator, not real network measurements.

| Bytes | Dense baseline | Compact uploads | Reduction |
|---|---:|---:|---:|
| Client model uplink | 61,554,618 | 10,513,014 | 82.92% |
| Total modeled application messages | 373,790,406 | 322,748,802 | 13.66% |

Total includes the unchanged modeled reference, BN, handoff and server traffic.
The model state fingerprints match after training. Compact round-boundary resume
matches uninterrupted compact training, including the communication ledger.
Measured source data: `output/denice_adaptive_tests/communication_comparison.json`.

## Checks

- Codec unit tests: 3 passed, including exact aggregation with zero-valued
  contributors, frozen coordinates, stale binding and malformed mask rejection.
- Existing P1 aggregation tests: 31 passed, 25 subtests passed.
- Hierarchical native training suite: 5 passed in 459.48 seconds, including
  compact/dense equivalence and both resume paths. Across suites: 39 tests passed.
- Python compilation of changed production modules and CLI: passed.
- `git diff --check`: passed.

Initial hierarchical test setup could not write into Windows sandbox temporary
directories. Rerun uses `tempfile.tempdir` under the writable repository output.
Python also emits an access-violation diagnostic while importing the existing
optional pyarrow dependency; codec tests nevertheless finish with exit code 0.
No dependency changes or skipped assertions were used to obtain these results.

Reproduction from repository root:

```text
rtk proxy python -m pytest tests/test_denice_compact_transport.py tests/test_denice_v2_p1.py -q
rtk proxy python -c "import tempfile,pathlib,pytest; p=pathlib.Path('output/denice_adaptive_tests').resolve(); p.mkdir(parents=True,exist_ok=True); tempfile.tempdir=str(p); raise SystemExit(pytest.main(['tests/test_hierarchical_cluster.py','-q']))"
```

The fixture is intentionally small. Training speed, memory overhead, full-dataset
accuracy, classification macro-F1 and routing macro-F1 improvements remain
unmeasured. The adaptive client/growth/router mechanisms are specifications,
not implemented or validated by these tests.
