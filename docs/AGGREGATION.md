# Aggregating evaluation results

One offline command exports overall quality, coverage, operation/target/criteria profiles and the operation × target grid. It requires no provider credentials or image downloads.

```bash
python scripts/aggregate_scores.py \
  --models example \
  --judge-tag gpt-4o \
  --mode all
```

Use `--output-root` to select model outputs, `--data-root` for the release metadata, and `--results-dir` to choose the export directory. The default is `results/<judge_tag>/<mode>/<denominator>/`. Repeating the command replaces files in that directory; retain a separate results directory for each experiment. Multiple tags may follow `--models`. The model directories must exist.

## Output files

| File | Contents |
| --- | --- |
| `summary.csv` | Overall Q (`score`), IF/VQ/DP, coverage at 0.6 and sample-status counts |
| `sample_scores.csv` | Full sample IDs, task, operation, target, status, inclusion and Q |
| `sample_criteria.csv` | Applicable sample-level criterion and group scores |
| `subtask_scores.csv` | Original subtask means; 64 rows per model in all-task mode |
| `operation_scores.csv` | Operation means; 16 rows per model in all-task mode |
| `target_scores.csv` | Final target means; eight rows per model in all-task mode |
| `criteria_scores.csv` | Eight criteria and three criterion-group means |
| `operation_target_scores.csv` | The 16 × 8 grid in all-task mode; empty combinations have blank scores |
| `aggregation_metadata.json` | Protocol, denominator choice, metadata/result hashes and judge information |

All profiles average sample-level values, rather than taking an unweighted average of subtask means. Criterion profiles include only samples where the criterion applies. Original task IDs determine subtask coverage; final target assignments determine target and cross-axis groups. These are different groupings and must not be interchanged.

A blank score is N/A, not zero. `not_applicable` means that a group contains no applicable samples; `no_included_samples` means it has samples but none qualify under the selected denominator. `incomplete` means unresolved results remain. In `gen`/`edit` modes, the scope and coverage denominator are restricted to the selected subtasks, not the full benchmark.

## Denominators and status

The policy is explicit and applies to every selected model:

- `--denominator available` (default): retain valid scored samples and explicitly unsupported tasks, with unsupported tasks scored as zero. Confirmed generation or judge execution failures are excluded. Missing, invalid or stale results are excluded but keep the summary marked incomplete. Report the included counts and policy because exclusions can raise the apparent score.
- `--denominator fixed`: use all expected samples in scope. Missing, invalid, stale and explicitly failed samples contribute zero.

Unsupported samples contribute zero under both policies. Inapplicable criteria never contribute zero. Coverage is the proportion of original subtask means at least 0.6. If any subtask has no included samples, coverage is blank rather than interpreting an unknown mean as a failure.

Missing files do not prove unsupported capability or execution failure. The script distinguishes `missing_output` from `missing_evaluation`. It recognizes unsupported records written by the evaluator. For other confirmed cases, an optional `--status-file` supplies explicit declarations:

```json
{
  "example": {
    "gen_animal_000": {
      "status": "execution_failed",
      "reason": "Inference terminated before producing an output."
    },
    "images/local_remove_portrait_face_013.jpg": {
      "status": "unsupported_capability",
      "reason": "This model does not support editing."
    }
  }
}
```

Only these two declared statuses are accepted, with a non-empty reason. IDs must exist in the dataset. A declaration conflicting with an existing evaluation record is rejected; it cannot silently replace measured scores. Declarations outside a selected gen/edit scope do not contribute to that run.

This release does not embed historical model-name exceptions or private experiment-specific error overrides. Its two policies make denominator choices visible; neither is a claim that every historical paper table is reproduced without its original status decisions. No tier thresholds are introduced in this batch.

## Record checks

The reader checks the release protocol, question and prompt hashes, sample ID, task, seed and complete question/criterion assignments. It recomputes criterion/group/Q scores from atomic answers instead of trusting cached summary values. Malformed records are classified as invalid, and incompatible records as stale. `--judge-model` optionally requires an exact judge model name; mixing judge models under the same result set is rejected.

The current evaluator does not hash output image contents or endpoint configuration. Use separate experiment tags or rerun evaluation when those change. The aggregator cannot infer that an apparently current record belongs to a replaced image.

## Offline verification

```bash
python -m unittest discover -s tests -v
```

Tests cover both denominator policies, unsupported samples, missing records, N/A cross-axis cells, sample-weighted means and invalid/stale records. They use temporary result fixtures and make no API calls.
