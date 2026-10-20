# MoI-Bench

### A Benchmark for Image Generation and Editing on Mobile Devices

[Project page](https://carlofkl.github.io/MoI-Bench-project/)

MoI-Bench evaluates image generation and editing through requests people make on mobile devices. It contains **1,150 samples across 64 subtasks**, organized by **16 operations** and **8 target categories**.

**MoI-Eval** uses sample-specific questions to assess instruction following, visual quality and detail preservation. Alongside overall quality, it reports capability profiles by operation, target and criterion. A separate Quality–Deployability map relates reference-model quality to measured mobile latency, storage and memory costs.

![Overview of MoI-Bench and MoI-Eval](assets/overview.webp)

## What can you evaluate?

- **Generation:** create images from text prompts.
- **Local editing:** add, remove, replace or adjust selected content.
- **Global editing:** change style, background or viewpoint, and restore image quality.
- **Understanding editing:** perform semantic edits, outpainting and inpainting.
- **Reference editing:** use one reference, or a base image with up to three references.

Targets include concrete and abstract objects, animals, body and limbs, face and hair, clothing and accessories, multiple subjects, and text and documents. Applicable operation–target combinations define the 64 subtasks; the benchmark is not a fully populated Cartesian product.

## Release checklist

This package includes the benchmark metadata and tools for downloading, browsing, evaluating and aggregating results.

- [x] Project website and qualitative examples
- [x] Final sample-level target mapping ([format](docs/DATA_FORMAT.md))
- [x] Benchmark metadata, instructions and task taxonomy
- [x] Data download and validation scripts, with image-source attribution
- [x] Reviewed evaluation rubrics and sample-level questions
- [x] Model-output format documentation ([examples](docs/OUTPUT_FORMAT.md))
- [x] MoI-Eval output judging ([usage](docs/EVALUATION.md))
- [x] Unified score-aggregation script ([usage](docs/AGGREGATION.md))
- [x] Unified operation, target, criteria and operation–target score exports
- [x] Read-only dataset browser

The current package focuses on data access and evaluation. Inference adapters are deferred; plotting and annotation/review tools are excluded.

## Using the benchmark

> **Current status:** data download, validation, browsing, output evaluation and aggregation commands are available. See the [release plan](docs/RELEASE_PLAN.md).

### 1. Get the repository and download the data

```bash
git clone https://github.com/Carlofkl/MoI-Bench.git
cd MoI-Bench
```

Use Python 3.10 or newer. Install the image-processing dependency and provide your own Pexels API key. The downloader reconstructs benchmark inputs from the recorded source metadata.

```bash
python -m pip install -r requirements.txt
export PEXELS_API_KEY="YOUR_PEXELS_API_KEY"

# Validate metadata without downloading images.
python scripts/validate_dataset.py

# Inspect planned downloads first.
python scripts/download_images.py --dry-run

# Download and reconstruct the benchmark inputs.
python scripts/download_images.py

# Validate metadata, image paths and dimensions.
python scripts/validate_dataset.py --verify-images
```

Use the released image-source and license records for attribution. Keep API keys outside version control.

### Browse the dataset

```bash
python scripts/browse_moi.py
# Optional Chinese interface:
python scripts/browse_moi.py --language zh
```

The read-only browser supports operation/target filters and prompt or sample-ID search. It uses the final labels in `targets.json`. Download images first to view them; instructions remain browsable without images. No review or annotation controls are included.

### 2. Prepare model outputs

Run your model using its own inference implementation. Model inference scripts are outside the current release scope. Follow the [output-format examples](docs/OUTPUT_FORMAT.md) to arrange existing outputs and declare unsupported tasks.

Outputs follow this layout:

```text
output/<model_tag>/<full_task>/<sample_directory>/
├── prompt.json
└── seed_0.png
```

`prompt.json` links each output to its benchmark entry and ordered input images. MoI-Eval evaluates one output per sample, using `seed_0.png`. A custom model should follow the same output contract and declare unsupported input configurations.

### 3. Evaluate output quality

The evaluator requires the released sample-level questions and credentials for the selected judge. For an OpenAI-compatible evaluation using GPT-4o:

```bash
python -m pip install -r requirements-eval.txt
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

python scripts/eval_outputs.py \
  --model-tag dreamlite \
  --judge openai \
  --judge-model gpt-4o \
  --judge-tag gpt-4o \
  --mode all
```

Judge calls incur provider charges. Keep the judge configuration fixed when comparing models. Questions use the same `0`, `0.6` and `1` scoring anchors; unsupported tasks must remain distinguishable from execution failures and missing outputs.

See [evaluation instructions](docs/EVALUATION.md) for output metadata, dry-run, capability declarations and resuming.

### 4. Aggregate and inspect results

```bash
python scripts/aggregate_scores.py \
  --mode all \
  --judge-tag gpt-4o \
  --models dreamlite
```

The aggregation command exports overall quality, task coverage, operation, target and criteria profiles, and operation–target scores in one run, without additional judge calls. These exports require sample-level scores and the released task/target metadata; overall scores alone are insufficient. The default uses the fixed benchmark denominator. Missing or invalid results retain their status and mark summaries as incomplete. See the [aggregation guide](docs/AGGREGATION.md) for denominator options and explicit failure declarations. Quality scores alone do not establish on-device performance: latency, storage and peak memory require separate measurements on the target device.

## Repository layout

```text
data/                   # Samples, questions, rubrics and target labels
  images/               # Image source records and reconstruction settings
scripts/                # Five runnable commands
  moi/                  # Shared dataset-reading and scoring modules
docs/                   # Data/output formats and usage guides
tests/                  # Offline checks
```

## License

Original project code, documentation and benchmark annotations are released under the [MIT License](LICENSE). Third-party images are not included or relicensed; use the download script and reconstruction settings under their original terms. See [image licensing](IMAGE_LICENSE.md).

## Citation

The BibTeX entry will be added when the publication metadata is finalized.
