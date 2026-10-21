# Model output format

Run your image model with its own inference code, then arrange its outputs as below. The evaluator reads instructions and ordered inputs from `data/samples.json`; it does not require model-specific inference wrappers.

```text
output/<model_tag>/<task>/<sample_directory>/
├── prompt.json
└── seed_0.png
```

- `model_tag`: your experiment name, for example `my_model_fp16`.
- `task`: the complete task key in `samples.json`, including its slashes.
- `sample_directory`: the sample-ID filename stem; see examples below.
- `seed_0.png`: the single image evaluated for this sample. This is a naming convention, not a requirement that your model use random seed zero.
- `prompt.json`: metadata linking the output to the full dataset ID and ordered inputs.

`output_sizes.json` is not required by the evaluation package. It may be kept as optional experiment metadata.

## Saved output size

- Text-to-image generation uses the model's default resolution. Wallpaper prompts use a 9:16 aspect ratio.
- An editing output must have the same pixel dimensions as its base image unless the instruction specifies another canvas size.
- Outpainting defaults to a 1:1 output canvas. An aspect ratio or pixel size stated in the instruction takes precedence.
- Saved outputs should not be resized after inference. Evaluation-time image preprocessing does not change the saved output.

## Text-to-image

Directory: `output/my_model/gen/animal/gen_animal_000/`

```json
{
  "entry_id": "gen_animal_000",
  "input_images": []
}
```

## Single-reference generation (R1)

Directory: `output/my_model/ref/r1/animal/ref_r1_animal_000/`

```json
{
  "entry_id": "images/ref_r1_animal_000.jpg",
  "input_images": ["images/ref_r1_animal_000.jpg"]
}
```

The sole input is the reference image. The R1 task has no additional base image.

## Base plus three references (B1+R3)

Directory: `output/my_model/ref/b1r3/mixed/ref_b1r3_mixed_000/`

```json
{
  "entry_id": "images/ref_b1r3_mixed_000.jpg",
  "input_images": [
    "images/ref_b1r3_mixed_000.jpg",
    "images/ref_b1r3_mixed_000_ref01.jpg",
    "images/ref_b1r3_mixed_000_ref02.jpg",
    "images/ref_b1r3_mixed_000_ref03.jpg"
  ]
}
```

Keep image order unchanged: input 1 is the base; inputs 2–4 are the references in manifest order. Ordinary editing uses the same format with only the base image; B1+R1 and B1+R2 append one and two references.

Do not replace the full `entry_id` with its filename stem. Paths in `input_images` are relative to `data/`, not the output directory. The evaluator rejects mismatched IDs or input order. If `input_images` is omitted, it uses the manifest order; including it is recommended for checking your inference pipeline.

## Unsupported tasks and failures

Do not invent placeholder outputs. Keep unsupported tasks separate from failed runs. The evaluator can mark unsupported multi-image edits using `--multi-image-support no`. Other confirmed unsupported tasks or execution failures can be declared for aggregation using the [status-file format](AGGREGATION.md#denominators-and-status).

An absent output alone does not establish unsupported capability. Likewise, an API judge failure does not establish zero image quality. Result tables retain these distinctions.

Use a new model tag when changing your checkpoint, precision or inference configuration, so records from different experiments do not share a directory.
