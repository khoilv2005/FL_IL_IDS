# Audit of 72 missing continuous profiles — 2026-10-05

## Decision

All **72 pairs across 50 clients** fall in **A: inherited binary memory for a
pre-join task**, under the recorded training history. Counts: A=72, B=0, C=0,
D=0, unresolved=0. Do not create retrospective local profiles for these pairs
by assuming that inherited class mappings establish historical data access.

## Sources and method

- Original results (4).zip, training commit `03b9b53`.
- Complete `denice_debug_history.json`: six task starts, 120 round records,
  bootstrap events and per-task client-data entries.
- `checkpoint_task_5_round_19.pt`: complete client algorithm metadata (no weight
  reconstruction is needed to inspect this metadata).
- Completed `denice_matched_oracle_diagnostics.zip`: 588-pair lifecycle table.
- Original runner source inspected with `git show 03b9b53:...` to establish the
  historical bootstrap contract, not merely the behavior of latest main.

Each of the 72 pairs meets all of the following:

1. It has local class support and binary memory in the final detector, but no
   retrospective continuous profile.
2. The task predates the client's first recorded local task.
3. The later creation event is `representative_clone` with a recorded donor.
4. The client is absent from that earlier task's client-data record.
5. The task is absent from the client's CGoFed completed-task records and is not
   its last local router refresh task.

The original bootstrap implementation copies the representative model and
ContextDetector, clears raw reference inputs, and resets the new client's
CGoFed projection state to an empty task history. The observed coverage pattern
therefore matches intentional inheritance rather than missing persistence.

Examples:

| Client | Missing tasks | First clone task | Donor |
| --- | --- | ---: | ---: |
| 35 | T0–T4 | 5 | 85 |
| 43 | T0–T3 | 4 | 85 |
| 85 | T0–T2 | 3 | 20 |
| 6 | T0–T1 | 2 | 5 |
| 1 | T0 | 1 | 15 |

Rejoining catch-up events are preserved separately in the per-pair output. They
do not establish that a client trained a task before its initial creation.

## Raw-row limitation

The original current local NPZ shards were not supplied to this local audit.
`local_shard_task_rows` and `available_rows` remain null, explicitly labelled
`not_read_original_shard_not_supplied`. No absence of current shard rows is
claimed. This does not prevent category A: the complete run history establishes
that these tasks precede the client's creation, independently of any rows a
current file might contain. The audit CLI accepts `--data-dir` to append direct
raw-label counts if desired; this does not automatically authorize profile fit.

No evidence in these 72 pairs identifies a category B provenance gap, category C
eligible-but-empty profile, or category D eligible fitting omission. This is a
conclusion about these pairs in this run, not an exhaustive persistence audit.

## Code correction and artifacts

Lifecycle `status` now derives from actual `continuous_profile_present`, not
`profile_evidence`. This fixes the misleading status rule without changing
Oracle predictions, route eligibility or any metric.

Reusable analysis command:

```
python tools/audit_denice_profile_provenance.py --results-zip "results (4).zip" --matched-zip denice_matched_oracle_diagnostics.zip --out audit_output
```

Local outputs under `C:\Users\khoak\Downloads\denice_profile_provenance_audit_20261005`:

- `profile_provenance_pairs.csv`: all requested pair fields, bootstrap source,
  evidence, row-count availability and A/B/C/D reason.
- `profile_provenance_summary.json`: counts and limitations.
- `profile_lifecycle_corrected.csv`: corrected display status for all 588 pairs.

The analysis command was executed on the supplied artifacts. No model training,
router fitting, new inference or unit test run was performed.

## Next gate

Keep `multiclass_balanced` as the frozen baseline after its exact integration
reproduction. Mahalanobis remains a secondary retrospective candidate; its
smaller permitted route bank is legitimate here. Do not claim that its missing
8.41 pp of allowed-route ceiling is recoverable by fixing lost profiles.

Proceed next to a separately scoped peer-assisted coverage diagnostic using
actual graph/protocol-compatible peers. Sharing binary descriptors at compatible
bootstrap time does not justify averaging continuous statistics across diverged
feature spaces. No top-k, peer coverage or full training was run by this audit.
