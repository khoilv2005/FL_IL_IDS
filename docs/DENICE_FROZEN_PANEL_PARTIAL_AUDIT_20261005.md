# Incomplete frozen-panel artifact audit — 2026-10-05

Input: `C:\Users\khoak\Downloads\denice_frozen_panel_diagnostics.zip`.
Evaluation commit: `cfe6c0caf7d89ebf926d14988d964b1337c334cf`.
This is the failed truncated-reference run, **before** batch replay fix `d27fd1f`.
`frozen_panel_completion.json` is false. No fresh-panel accuracy is available.

## Protocol and panel integrity

- Protocol bytes match the SHA256 in `frozen_panel_lock.json`.
- Checkpoint 03b9b53 task 5 round 19; k=16 plus self; candidate seed 42;
  panel seed 20261006; partition seed 523687; batch size 512.
- Graph validation passes: all 8,320 positive edge IDs and round-trip weights
  exactly match the checkpoint. Historical default-parser feature weights are retained.
- `panel_manifest.csv` contains 50,000 rows, 50,000 distinct row IDs and 50,000
  distinct content hashes, assigned to 98 receivers with 510–511 samples each.
- Independently compared against original V2 predictions: **zero old row-ID overlap**.
- Independently compared against all V2 provenance roles: **zero gate-role hash overlap**.
- Old-panel content overlap is recorded as zero by the runtime guard. This separate
  local audit has no complete raw test NPZ and therefore does not independently
  re-hash those old inputs or re-prove the dataset's original train/test split.
- Recorded global test NPZ SHA256:
  `bfdb96f41a233cbc5a574074899c98cd5dd766bb6b109e8756ce328bcd32e54f`.

## Actual scope

Present classes: 1, 2, 4–27, 29, 32 (28 classes).
Missing classes: **0, 3, 28, 30, 31, 33**.
All six have zero remaining row IDs: the old panel exhausted their entire pools.
Their raw test counts are respectively 923, 1,550, 648, 1,507, 359 and 1,111.

Class 2 has only 121 remaining row IDs; 3 were rejected for forbidden content,
leaving **118 samples**. Other present classes have 1,847 or 1,848 samples each.
Only class 32 represents T5. A T5 metric on this panel cannot describe all four
original T5 classes. The user-approved remaining-test scope must be reported explicitly.

For context only, restricting the **old development panel** to these 28 classes
leaves 43,902 samples and gives:

| Old policy on present-class subset | Accuracy % |
| --- | ---: |
| Self | 31.2537 |
| Majority k=16 | 48.7882 |
| Gate V2 k=16 | 52.3006 |
| Frozen class meta k=16 | 56.5008 |

These are old subset metrics, not fresh-panel results. The subset also has different
class frequencies, particularly for class 2; it is not a paired or directly matched
distribution comparison. Full-34-class 50.732% should not be used as an interchangeable
benchmark for this panel. Compare frozen policies within the same fresh panel first.

## Failure evidence and correction

- 37 donors completed. Recorded forward rows total 323,113 (including reference).
- 614 fresh feature NPZs contain 313,289 expert-sample predictions. They are partial
  expert-query records, not 313,289 unique test samples or complete panel decisions.
- No summary, per-client policy metrics or final prediction CSVs were produced.
- Failure logged at receiver 12 / donor 30 / class_margin. In the old comparison
  order, prediction/task fields for that reference pair passed before margin failed.
- This ZIP contains neither `reference_reproduction.json` nor actual/expected failure
  arrays. It cannot determine the failing margin value, row or error magnitude.
  Numerical batch drift is plausible, but is not quantified by this artifact.

Fix `d27fd1f` replaces truncated references mixed with fresh inputs with a separately
executed complete old receiver stream at batch 512. It preserves model/scaler/router,
sampling seeds, decision policy and all comparison tolerances. It adds explicit
failure arrays and error statistics. Fresh-panel performance remains unmeasured
until the revised reference checks and all expert queries complete. Do not tune
the selector using partial data or relax the guard based only on this error message.
