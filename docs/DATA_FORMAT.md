# Dataset format

## Target mapping (available)

`data/targets.json` contains the final target assignments for 1,150 samples. The metadata files below are available, together with download, validation, browsing, evaluation and aggregation scripts.

The `items` object is keyed by the release sample ID. Generation IDs use `gen_<target>_<index>`, for example `gen_animal_000`, omitting the redundant operation component. Other sample IDs are unchanged. All future sample manifests, question records and output references must use the same release IDs. Each item contains:

| Field | Meaning | Example |
| --- | --- | --- |
| `task` | Original benchmark task key | `gen/animal` |
| `operation_family` | Operation family | `gen` |
| `operation` | Stable operation identifier | `gen` |
| `operation_label` | Paper/figure display label | `T2I` |
| `target_group` | Broad target group | `animal` |
| `target` | One of the eight target identifiers | `animal` |
| `target_label` | Target display label | `Animals` |
| `assignment_source` | Original assignment provenance | `existing_specific_task_label` |

`target_group_counts` and `target_counts` report the sample counts. `provenance` records the original source hash and its historical benchmark/decision hashes; these are not hashes of the renamed release files. Historical source paths and schema names are retained only for traceability.

### Migration from development fields

| Development field | Release field |
| --- | --- |
| `behavior_family` | `operation_family` |
| `behavior` | `operation` |
| `content_major` | `target_group` |
| `content_key` | `target` |
| `content_label` | `target_label` |
| `major_counts` | `target_group_counts` |
| `detailed_counts` | `target_counts` |
| `n_prior_cross_content_decisions` | `provenance.n_prior_cross_target_decisions` |

Display names use `Face & Hair` and `Multiple Subjects`. Operation labels use T2I, Addition, Removal, Replacement, Adjustment, Stylization, Background, Restoration, Viewpoint, Semantic, Outpainting, Inpainting, R1, B1+R1, B1+R2 and B1+R3. Identifiers such as `global/camera` remain stable; use `operation_label` for display rather than deriving identifiers from display names. Task paths are unchanged; generation sample IDs follow the normalization above.

The mapping supplies statistical labels, not new benchmark tasks. Operation–target groups may combine or split original subtasks. Keep original subtask IDs when calculating coverage. No target assignment or score was changed in this migration. An old generation ID is converted by removing its redundant second `gen_` component; legacy result readers will need this compatibility conversion.

## Additional migrated files

| File under `data/` | Contents |
| --- | --- |
| `taxonomy.json` | 64 original subtask definitions and counts, extracted from the final rubric records and checked against the sample manifest |
| `samples.json` | 1,150 instructions and ordered input references |
| `rubrics.json` | Eight criterion definitions and task-level applicability |
| `questions.jsonl` | 1,150 sample records containing 13,897 approved questions |
| `images/manifest.csv` | 1,130 image records, hashes and dimensions |
| `images/sources.csv` | Original image sources, authors and licence information |
| `images/derivations.json` | Source-image reconstruction settings |

Actual image files are not included in this migration. Paths in samples and image CSVs are relative to `data/`; derivations retain `path_base: ".."` relative to their own directory. Question evidence paths are relative to the repository root.

`samples.json` intentionally excludes obsolete embedded questions, weights and question-generation flags. Only `questions.jsonl` supplies the final sample-specific questions, applicability and group weights. Generation IDs and their question-ID prefixes omit the repeated `gen_`. Editing sample IDs and image filenames remain unchanged.

### Evaluation terminology

| Development identifier | Release identifier |
| --- | --- |
| `IF-InstructionFidelity` | `IF-Fidelity` |
| `IF-TargetScope` | `IF-Selection` |
| `EQ-Integrity` | `VQ-Integrity` |
| `EQ-Coherence` | `VQ-Coherence` |
| `EQ-Presentation` | `VQ-Presentation` |
| `DP-Context` | `DP-Context` |
| `DP-Structure` | `DP-Structure` |
| `DP-Subject` | `DP-Identity` |

Fields use `criterion`, `criteria`, `available_criteria`, `active_criteria`, `criterion_applicability`, `group_weights` and `criterion_aggregation`. Group `EQ` becomes `VQ`. Task-level `task_target_group` and `task_target_subcategory` describe original scopes, which may be `all` or `mixed`; sample-level labels come from `targets.json`. The source CSV column `subcategory` is renamed `task`.

Question text, scoring anchors, group-weight values and applicability decisions are preserved. Review interface state, generator metadata and reviewer notes are omitted from the release question records; the source retains this audit history. All 1,150 source records were approved with no pending-review flag.

These are new release schemas. Renamed IDs and fields change record hashes; historical judge results must pass an explicit compatibility conversion before reuse. Do not disable hash validation or claim byte-level compatibility with the development files. A local migration report records the identifier mappings and file hashes for implementing that conversion later.

### Runtime-focused rubrics

`rubrics.json` retains English criterion definitions, scoring boundaries and applicability guidance. Chinese translations of these descriptions, question budgets, question blueprints and authoring facets are omitted. These fields were used to design questions, not to evaluate fixed benchmark outputs. Task-level weights are rubric defaults; the final sample-specific applicability and weights in `questions.jsonl` are authoritative for scoring. No questions are generated during benchmark evaluation.
