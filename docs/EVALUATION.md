# Evaluating model outputs

The evaluator uses the fixed questions in `data/questions.jsonl`. It does not generate questions or run an image model. The current backend is OpenAI Chat Completions with image inputs and structured JSON output. Other providers must implement that interface to use the same adapter; compatibility with every custom endpoint is not assumed.

## Install and configure

```bash
python -m pip install -r requirements-eval.txt
export OPENAI_API_KEY="YOUR_API_KEY"
# Optional: set OPENAI_BASE_URL to your OpenAI-compatible endpoint.
```

## Prepare outputs

```text
output/<model_tag>/<task>/<sample_directory>/
├── prompt.json
└── seed_0.png
```

`task` is the complete slash-separated key from `samples.json`. `sample_directory` is the filename stem of the full sample ID: `gen_animal_000` for generation, or `local_remove_portrait_face_013` for `images/local_remove_portrait_face_013.jpg`. This directory convention does not replace the full sample ID used to join metadata.

A generation `prompt.json` can contain:

```json
{"entry_id": "gen_animal_000", "input_images": []}
```

For editing, `entry_id` must be the full `images/...jpg` ID. `input_images` lists that input first, followed by every `reference_images` entry in manifest order. R1 has one reference input; B1+R3 has the base and three reference inputs. Inputs are resolved relative to the samples file's declared path base. Model inference scripts are not included.

## Preview and run

```bash
python scripts/eval_outputs.py --model-tag example --judge openai \
  --judge-model gpt-4o --judge-tag gpt-4o --mode all --dry-run

python scripts/eval_outputs.py --model-tag example --judge openai \
  --judge-model gpt-4o --judge-tag gpt-4o --mode all
```

Dry-run does not require credentials, open images, write evaluation records or make API calls. The model output directory must exist. Only samples with both `prompt.json` and `seed_0.png` are scheduled for judging; the evaluator does not interpret missing outputs as unsupported tasks. Invalid metadata is reported and skipped. Use `--limit`, `--task` or `--entry-id` for a small first run. `--workers` controls concurrent samples; questions within a sample are evaluated sequentially. Each question produces an API request and may incur charges.

The optional `--multi-image-support no` explicitly marks multi-image edits unsupported and writes zero-valued question scores without contacting a judge. `yes` enables normal evaluation. `auto` reads capability declarations from `output/<model_tag>/_meta/*.json` (`supports_multi_image_edit`, optionally `maximum_input_images` and `supported_multi_image_task`). When capability is unknown, present outputs are evaluated and absent outputs remain missing. This option concerns multi-image editing only; it does not declare generation-only or editing-only support.

For R1, the prompt identifies the sole input as a reference image rather than a base image. This image-role correction is included in the prompt fingerprint, so prior records with the older prompt are not reused.

The judge receives ordered input images followed by the output, the instruction, one question and its three scoring anchors. Images are resized to a maximum side of 1,024 pixels and JPEG-encoded at quality 88, preserving the original evaluator's preprocessing. Valid scores are 0, 0.6 and 1. Malformed responses and API errors are failures, not zero-quality outputs.

## Results and resuming

Records are written to `output/<model_tag>/eval/<judge_tag>/<task>/<sample_directory>/seed_0.json`. They include question-level scores, `criterion_scores`, `group_scores`, normalized group weights and `usable_score` (sample Q), plus prompt and question hashes. Unsupported records have a separate status. Errors are reported and the command exits nonzero; failed samples do not receive fabricated scores.

Scoring averages questions within each applicable criterion, then criteria within IF/VQ/DP, then combines applicable group means using a weighted geometric mean. Inapplicable criteria are omitted. Applicable group weights are renormalized.

The release record version is `moi-eval-v1`. It preserves the scoring formula, but renamed IDs and fields mean historical records are not directly reusable. No legacy-result conversion is included in this batch. Existing matching records are skipped; `--force` reruns them. Use a new model/judge tag or `--force` if you change output images, endpoint or inference settings, since image-content and endpoint changes are not part of the resume check.

Use `aggregate_scores.py` for cross-sample results; see [AGGREGATION.md](AGGREGATION.md) for denominator and status rules. This evaluator does not assign overall quality tiers.

## Offline checks

```bash
python -m unittest discover -s tests -v
```

These tests do not call a provider. Live endpoint compatibility and judged image quality are not established by offline checks.
