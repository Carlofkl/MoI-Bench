#!/usr/bin/env python3
"""Validate finalized MoI-Bench metadata and optional local image files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_RESTORE_REQUIREMENTS = {
    "images/global_restore_object_008.jpg": "add_motion_blur",
    "images/global_restore_object_012.jpg": "add_moire_pattern",
    **{
        f"images/global_restore_object_{index:03d}.jpg": "add_sensor_noise"
        for index in range(15, 20)
    },
    **{
        f"images/global_restore_object_{index:03d}.jpg": "add_motion_blur"
        for index in range(20, 25)
    },
    **{
        f"images/global_restore_object_{index:03d}.jpg": "add_moire_pattern"
        for index in range(25, 30)
    },
}

REFERENCE_COUNTS = {
    "r1": 0,
    "b1r1": 1,
    "b1r2": 2,
    "b1r3": 3,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--verify-images", action="store_true",
        help=(
            "Require every image, decode it, and verify its dimensions and format. "
            "SHA-256 differences are warnings unless --strict-hash is used."
        ),
    )
    parser.add_argument(
        "--strict-hash", action="store_true",
        help=(
            "Treat image SHA-256 differences as errors during --verify-images. "
            "Use this for archival byte-for-byte checks."
        ),
    )
    args = parser.parse_args()
    if args.strict_hash and not args.verify_images:
        parser.error("--strict-hash requires --verify-images")
    return args


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    args = arguments()
    root = args.data_root.resolve()
    errors: list[str] = []
    hash_mismatches: list[tuple[str, str, str]] = []

    review = json.loads((root / "samples.json").read_text(encoding="utf-8"))
    items = review.get("items") or {}
    observed_tasks = Counter(record.get("task", "") for record in items.values())
    taxonomy = json.loads((root / "taxonomy.json").read_text(encoding="utf-8"))
    expected_tasks = {task: row["n_samples"] for task, row in taxonomy["subtasks"].items()}
    if dict(observed_tasks) != expected_tasks:
        for task in sorted(set(observed_tasks) | set(expected_tasks)):
            if observed_tasks[task] != expected_tasks.get(task, 0):
                errors.append(
                    f"task count {task}: dataset={observed_tasks[task]}, "
                    f"taxonomy={expected_tasks.get(task, 0)}"
                )

    targets_doc = json.loads((root / "targets.json").read_text(encoding="utf-8"))
    targets = targets_doc["items"]
    rubrics = json.loads((root / "rubrics.json").read_text(encoding="utf-8"))
    questions = [json.loads(line) for line in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    question_ids = [row["sample_id"] for row in questions]
    if len(question_ids) != len(set(question_ids)):
        errors.append("duplicate sample_id in questions.jsonl")
    if set(items) != set(targets) or set(items) != set(question_ids):
        errors.append("sample IDs differ between samples, targets and questions")
    if set(expected_tasks) != set(rubrics["subtasks"]):
        errors.append("subtask IDs differ between taxonomy and rubrics")
    if Counter(row["target"] for row in targets.values()) != targets_doc["target_counts"]:
        errors.append("target counts do not match target assignments")
    all_question_ids = []
    for row in questions:
        sid = row["sample_id"]
        sample = items.get(sid)
        if sample is None:
            continue
        if row["task"] != sample["task"] or targets.get(sid, {}).get("task") != sample["task"]:
            errors.append(f"task mismatch: {sid}")
        if row["instruction"] != sample["instruction"]:
            errors.append(f"instruction mismatch: {sid}")
        active = set(row["active_criteria"])
        if not active <= set(rubrics["criteria"]):
            errors.append(f"unknown active criterion: {sid}")
        if set(q["criterion"] for q in row["questions"]) != active:
            errors.append(f"question criteria do not match active criteria: {sid}")
        if not set(row["group_weights"]) <= {"IF", "VQ", "DP"}:
            errors.append(f"unknown criterion group: {sid}")
        for question in row["questions"]:
            all_question_ids.append(question["question_id"])
            if question.get("score_values") != [0, 0.6, 1]:
                errors.append(f"invalid score anchors: {question['question_id']}")
    if len(all_question_ids) != len(set(all_question_ids)):
        errors.append("duplicate question_id in questions.jsonl")

    used_paths: list[str] = []
    for sample_id, record in items.items():
        task = str(record.get("task") or "")
        if not record.get("instruction", "").strip():
            errors.append(f"empty English instruction: {sample_id}")

        is_image_sample = sample_id.startswith("images/")
        is_generation = task.startswith("gen/")
        if is_generation and is_image_sample:
            errors.append(f"generation sample unexpectedly has a base image: {sample_id}")
        if not is_generation and not is_image_sample:
            errors.append(f"editing sample is missing its base image: {sample_id}")
        if is_image_sample:
            used_paths.append(sample_id)

        references = record.get("reference_images") or []
        if not references and record.get("reference_image"):
            references = [record["reference_image"]]
        if task.startswith("ref/"):
            reference_mode = task.split("/", 2)[1]
            expected_references = REFERENCE_COUNTS.get(reference_mode)
            if expected_references is None:
                errors.append(f"unknown reference-edit mode {reference_mode}: {sample_id}")
            elif len(references) != expected_references:
                errors.append(
                    f"reference count {sample_id}: task={task}, "
                    f"expected={expected_references}, observed={len(references)}"
                )
        elif references:
            errors.append(
                f"non-reference task unexpectedly has {len(references)} "
                f"reference image(s): {sample_id}"
            )
        used_paths.extend(str(value) for value in references)
    if len(used_paths) != len(set(used_paths)):
        duplicates = [path for path, count in Counter(used_paths).items() if count > 1]
        errors.append(f"image paths reused unexpectedly: {duplicates[:5]}")

    licenses = csv_rows(root / "images" / "sources.csv")
    manifest = csv_rows(root / "images" / "manifest.csv")
    derivation_doc = json.loads(
        (root / "images" / "derivations.json").read_text(encoding="utf-8")
    )
    derivations = derivation_doc.get("items") or {}
    license_by_path = {row["local_path"]: row for row in licenses}
    manifest_by_path = {row["local_path"]: row for row in manifest}
    expected_path_set = set(used_paths)
    if set(license_by_path) != expected_path_set:
        errors.append("sources.csv path set does not match samples.json")
    if set(manifest_by_path) != expected_path_set:
        errors.append("manifest.csv path set does not match samples.json")
    if len(license_by_path) != len(licenses):
        errors.append("duplicate local_path in sources.csv")
    if len(manifest_by_path) != len(manifest):
        errors.append("duplicate local_path in manifest.csv")

    for path, required_operation in SYNTHETIC_RESTORE_REQUIREMENTS.items():
        operations = derivations.get(path, {}).get("operations") or []
        operation_names = {operation.get("operation") for operation in operations}
        if required_operation not in operation_names:
            errors.append(
                f"synthetic restoration {path}: missing {required_operation}"
            )
        if manifest_by_path.get(path, {}).get("reconstruction_mode") != "derived_jpeg":
            errors.append(
                f"synthetic restoration {path}: reconstruction_mode must be derived_jpeg"
            )

    required_license = ("provider", "source_id", "source_url", "license", "license_url")
    for path, row in license_by_path.items():
        missing = [field for field in required_license if not row.get(field, "").strip()]
        if missing:
            errors.append(f"license {path}: missing {', '.join(missing)}")

    if args.verify_images:
        try:
            from PIL import Image
        except ImportError:
            errors.append("Pillow is required for --verify-images")
        else:
            for index, (relative, row) in enumerate(manifest_by_path.items(), 1):
                path = (root / relative).resolve()
                if root not in path.parents or not path.is_file():
                    errors.append(f"missing image: {relative}")
                    continue
                observed_hash = digest(path)
                expected_hash = row["sha256"]
                if observed_hash != expected_hash:
                    hash_mismatches.append((relative, observed_hash, expected_hash))
                    if args.strict_hash:
                        errors.append(
                            f"SHA-256 mismatch {relative}: "
                            f"{observed_hash[:12]} != {expected_hash[:12]}"
                        )
                try:
                    with Image.open(path) as image:
                        observed_size = image.size
                        observed_format = image.format
                        image.load()
                except (OSError, ValueError) as exc:
                    errors.append(f"unreadable image {relative}: {exc}")
                    continue
                expected_size = (int(row["width"]), int(row["height"]))
                if observed_size != expected_size:
                    errors.append(
                        f"size mismatch {relative}: {observed_size} != {expected_size}"
                    )
                if observed_format != row["format"]:
                    errors.append(
                        f"format mismatch {relative}: {observed_format} != {row['format']}"
                    )
                if index % 250 == 0:
                    print(f"  verified {index}/{len(manifest_by_path)} images")

    if hash_mismatches and not args.strict_hash:
        print(
            f"WARNING: {len(hash_mismatches)} image(s) have different SHA-256 "
            "bytes; treating them as cross-environment encoding warnings.",
            file=sys.stderr,
        )
        for relative, observed, expected in hash_mismatches[:5]:
            print(
                f"- {relative}: {observed[:12]} != {expected[:12]}",
                file=sys.stderr,
            )
        remaining = len(hash_mismatches) - 5
        if remaining > 0:
            print(f"- ... and {remaining} more", file=sys.stderr)
        print(
            "  Use --strict-hash for archival byte-for-byte validation.",
            file=sys.stderr,
        )

    if errors:
        print(f"FAILED: {len(errors)} issue(s)", file=sys.stderr)
        for error in errors[:50]:
            print(f"- {error}", file=sys.stderr)
        return 1
    providers = Counter(row["provider"] for row in licenses)
    print(
        f"OK: {len(items)} samples, {len(observed_tasks)} subtasks, "
        f"{len(used_paths)} images, providers={dict(providers)}"
    )
    if args.verify_images:
        print("OK: all local images decode and match expected dimensions and formats")
        if not hash_mismatches:
            print("OK: all local image bytes match manifest.csv SHA-256 values")
        elif not args.strict_hash:
            print(
                f"OK with warnings: ignored {len(hash_mismatches)} SHA-256 "
                "difference(s)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
