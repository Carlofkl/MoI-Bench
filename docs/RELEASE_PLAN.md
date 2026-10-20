# Release scope and status

## Included

| Command | Purpose |
| --- | --- |
| `download_images.py` | Download and reconstruct input images |
| `validate_dataset.py` | Check metadata and optional downloaded images |
| `browse_moi.py` | Browse samples without changing data |
| `eval_outputs.py` | Evaluate existing model outputs |
| `aggregate_scores.py` | Export all score profiles in one run |

Shared modules are limited to dataset reading and scoring. The dataset includes task definitions, sample instructions, final target assignments, rubrics, fixed questions and source-image records.

## Excluded

Model inference adapters are deferred. Plotting scripts, annotation/review tools, question-authoring tools, private services and intermediate experiment outputs are not included.

## Verification

Metadata and offline tests are checked locally. Download dry-runs make no network requests. Live image downloads and paid judge requests have not been verified for this release package. The included tests are not measured model results.

## Remaining publication items

- MIT covers original project materials; see `IMAGE_LICENSE.md` for third-party images.
- Add the BibTeX entry when publication metadata is final.
- Complete live download and judge checks when credentials and test outputs are available.

See [data format](DATA_FORMAT.md), [output format](OUTPUT_FORMAT.md), [evaluation](EVALUATION.md) and [aggregation](AGGREGATION.md) for usage. Local preparation does not mean the files have been pushed to GitHub.
