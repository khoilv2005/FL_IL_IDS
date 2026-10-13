# APPLIANCE verification — 14 October 2026

## Completed checks

| Check | Result |
|---|---|
| `python -m pytest tests/test_denice_appliance.py tests/test_denice_transfer.py -q -p no:cacheprovider -p no:faulthandler` | **20 passed**, final run 12.16 seconds, exit 0 |
| Python compilation of entry point, protocol, modified runner and smoke tool | Passed, exit 0 |
| `git diff --check` | Passed |
| Direct runner: `python tools/smoke_denice_appliance.py` | Three tasks × two rounds × two clients completed, including router/task transitions, final evaluation and continuation/checkpoint writes |
| Kaggle entry-point subprocess: `python tools/smoke_denice_appliance.py --entrypoint --output output/appliance_entrypoint_verified` | All six rounds completed, exit 0 |

The unit suite checks deterministic election and ties; leader replacement; unavailable/empty groups; unequal cluster/sample weighting and unsupported coordinates; stale rounds/tasks/schema/versions and nonfinite parameters; private-bank bounds; accepted helpful transfer; harmful-transfer rejection; protected-row preservation; source packet loss; leader mailbox delivery counts; registration-only acceptance; exception/mode/RNG restoration; split disjointness/reproducibility and singleton retention; structural/exact feature fingerprints; and unsupported configurations. Existing full-model transfer tests remain passing.

The direct runner smoke produced 34 source class packets and **zero accepted transfers**. Its 48 receiver/class decisions were 6 `no_compatible_donor`, 18 `no_functional_gap`, and 24 `protected_row`. Summary: `output/appliance_smoke/summary.json`; detailed log: `output/appliance_smoke/training.log`.

The entry-point smoke produced 46 packets and **zero accepted transfers**. Its decisions were 2 `no_compatible_donor`, 22 `no_functional_gap`, and 24 `protected_row`. Summary: `output/appliance_entrypoint_verified/summary.json`; detailed log: `output/appliance_entrypoint_verified/training.log`. The entry point supplies additional native defaults compared with the standalone runner fixture, so these are integration checks, not a paired performance comparison. Explicit leader-mailbox audit counters were added afterward and covered by the final unit run; the full training smoke was not repeated solely for those counters.

Synthetic examples have six artificial classes and 16×1 inputs, 40 training-file records per class/client and a separately generated test. These are **not IDS data**. Synthetic accuracy is intentionally not presented as evidence for the method. A controlled two-client unit fixture demonstrates a successful beneficial row transfer; that does not establish that real DENICE feature bases will pass the compatibility gate.

## Findings and limits

The entry-point smoke exposed the inherited `resume_output_dir` taking precedence over an explicit `output_dir`. The delivered preset clears that value for fresh runs; the corrected entry point completed successfully. Fresh training is enforced because private validation banks and election history are not included in exact continuation state.

The strict class-wise CE constraints can reject registration even when target recall improves, because introducing an additional competing class can increase negative-class cross-entropy. This is intentional conservatism, and should be measured in a predeclared ablation. Do not weaken it in response to final-test performance.

Local environment: Windows, Python 3.12; PyTorch 2.5.1+cu121, NumPy 2.4.6, scikit-learn 1.9.0, pandas 3.0.3, pytest 9.1.1. An initial pytest run printed a PyArrow native-import access-violation diagnostic despite completing with exit 0; it also could not write the pre-existing pytest cache directory. The final command disabled the cache and faulthandler plugins and completed all 20 tests with exit 0. These options avoid those diagnostics; they do not repair the system's optional PyArrow installation. No system packages were modified.

Not executed: full 100-client/34-class IDS training; a run on Kaggle hardware; multi-seed scientific comparisons; inference from exported archives on Kaggle; real network failure/recovery; cryptographic secure aggregation, differential privacy or attack-resistance evaluation. Dependencies are minimum-version requirements rather than a reproducible lockfile. Keep an environment snapshot for actual experiments.

The source bundle is built by `tools/package_denice_appliance.py`; it verifies ZIP CRC and every member's SHA-256 against `SOURCE_MANIFEST.json`. It contains code/configuration/docs/tests, not raw datasets or trained models.
