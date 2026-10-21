"""Evaluate existing model outputs with the fixed MoI-Eval questions."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

from moi.dataset import DEFAULT_EVAL_JSON, input_image_paths, load_dataset, resolve_data_path
from moi.scoring import (
    EVAL_PROTOCOL_VERSION,
    SCORE_VALUES,
    aggregate_question_scores,
    normalize_score,
    zero_question_results,
)


DEFAULT_QUESTIONS_JSONL = REPO_ROOT / "data" / "questions.jsonl"
# ────────────────────────────────────────────────────────────────────────
# Prompt templates. Kept in-code so their SHA-256 can be stored per record
# and any future prompt change is traceable.
# ────────────────────────────────────────────────────────────────────────

SYSTEM_EDIT = (
    "You are a strict visual judge for an image-editing benchmark. "
    "Compare the ordered input image or images with the edited result and score "
    "only the stated atomic question. Use the supplied score anchors exactly. "
    "Do not reward unrelated visual quality and do not speculate beyond visible "
    "evidence. Output only one JSON object: {\"score\": 0}, "
    "{\"score\": 0.6}, or {\"score\": 1}."
)

USER_EDIT_TEMPLATE = (
    "{image_description}\n\n"
    "Editing instruction:\n"
    "\"\"\"\n{instruction}\n\"\"\"\n\n"
    "Question to answer:\n"
    "\"\"\"\n{question}\n\"\"\"\n\n"
    "Score 1 condition:\n{success_condition}\n\n"
    "Score 0.6 condition:\n{partial_condition}\n\n"
    "Score 0 condition:\n{failure_condition}\n\n"
    "Return only {{\"score\": 0}}, {{\"score\": 0.6}}, or {{\"score\": 1}}."
)

SYSTEM_GEN = (
    "You are a strict visual judge for a text-to-image benchmark. Inspect the "
    "generated image against the prompt and score only the stated atomic "
    "question. Use the supplied score anchors exactly. Do not speculate beyond "
    "visible evidence. Output only one JSON object: {\"score\": 0}, "
    "{\"score\": 0.6}, or {\"score\": 1}."
)

USER_GEN_TEMPLATE = (
    "The attached image is the GENERATED image.\n\n"
    "Generation prompt:\n"
    "\"\"\"\n{prompt}\n\"\"\"\n\n"
    "Question to answer:\n"
    "\"\"\"\n{question}\n\"\"\"\n\n"
    "Score 1 condition:\n{success_condition}\n\n"
    "Score 0.6 condition:\n{partial_condition}\n\n"
    "Score 0 condition:\n{failure_condition}\n\n"
    "Return only {{\"score\": 0}}, {{\"score\": 0.6}}, or {{\"score\": 1}}."
)

PROMPT_TEMPLATES_HASH = hashlib.sha256(
    (SYSTEM_EDIT + USER_EDIT_TEMPLATE + SYSTEM_GEN + USER_GEN_TEMPLATE + "image_roles_v2_r1_reference").encode()
).hexdigest()[:16]

# JSON schema for GPT structured output.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number", "enum": list(SCORE_VALUES)}},
    "required": ["score"],
    "additionalProperties": False,
}


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────


def resize_max(img, max_side: int = 1024):
    """Downscale so longest side == max_side; preserves aspect. Keeps small imgs."""
    from PIL import Image
    if max(img.size) <= max_side:
        return img
    w, h = img.size
    s = max_side / max(w, h)
    return img.resize((int(round(w * s)), int(round(h * s))), Image.BICUBIC)


def pil_to_base64(img, fmt="JPEG", quality=88) -> str:
    """Encode PIL image to base64 (used for GPT vision input)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_score_response(raw: str) -> float:
    """Parse only a protocol score; never turn an API/parser failure into zero."""
    text = str(raw or "").strip()
    try:
        value = json.loads(text)
        if isinstance(value, Mapping):
            return normalize_score(value.get("score"))
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    raise ValueError(f"judge did not return a valid score JSON: {text[:300]!r}")


# ────────────────────────────────────────────────────────────────────────
# Backend abstraction
# ────────────────────────────────────────────────────────────────────────


class Judge(ABC):
    name: str
    provider: str

    @abstractmethod
    def answer(self, system: str, user_text: str, images: List) -> Tuple[float, str]:
        """Return ``(score, raw_response_text)`` for a valid judge response."""


class OpenAIJudge(Judge):
    provider = "openai"

    def __init__(self, model: str = "gpt-4o", max_retries: int = 2,
                 request_timeout: int = 90):
        from openai import OpenAI  # type: ignore
        self.name = model
        self.client = OpenAI(timeout=request_timeout, max_retries=0)
        self.max_retries = max_retries

    def _build_messages(self, system: str, user_text: str, images: List) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for img in images:
            b64 = pil_to_base64(resize_max(img, 1024))
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    def answer(self, system: str, user_text: str, images: List) -> Tuple[float, str]:
        messages = self._build_messages(system, user_text, images)
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    # gpt-5 family uses max_completion_tokens
                    max_completion_tokens=40,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "moi_score",
                            "strict": True,
                            "schema": ANSWER_SCHEMA,
                        },
                    },
                )
                raw = resp.choices[0].message.content
                return parse_score_response(raw), raw
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"judge request failed after retries: {last_err}")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_questions(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"{path}:{line_number}: missing sample_id")
            if sample_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id}")
            if record.get("schema_version") != "moi-questions-v1":
                raise ValueError(f"{sample_id}: unsupported question schema")
            questions = record.get("questions")
            if not isinstance(questions, list) or not questions:
                raise ValueError(f"{sample_id}: approved record has no questions")
            for question in questions:
                for field in (
                    "question_id", "criterion", "question", "question_zh",
                    "success_condition", "partial_condition", "failure_condition",
                ):
                    if not str(question.get(field) or "").strip():
                        raise ValueError(f"{sample_id}: question missing {field}")
            records[sample_id] = record
    return records


def read_multi_image_capability(model_root: Path, override: str) -> Dict[str, Any]:
    """Read editing capability from inference metadata, with a CLI override."""
    if override in {"yes", "no"}:
        return {
            "known": True,
            "supported": override == "yes",
            "maximum_input_images": None,
            "supported_task": None,
            "source": f"cli:{override}",
        }

    candidates: List[Tuple[int, Path, Dict[str, Any]]] = []
    for path in sorted((model_root / "_meta").glob("*.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if "supports_multi_image_edit" not in metadata:
            continue
        priority = 0 if metadata.get("mode") == "edit" or "edit" in path.stem else 1
        candidates.append((priority, path, metadata))
    if not candidates:
        return {
            "known": False,
            "supported": None,
            "maximum_input_images": None,
            "supported_task": None,
            "source": None,
        }

    _, source, metadata = sorted(candidates, key=lambda row: (row[0], str(row[1])))[0]
    value = metadata.get("supports_multi_image_edit")
    maximum = metadata.get("maximum_input_images")
    if not isinstance(maximum, int):
        maximum = None
    if isinstance(value, bool):
        supported: Any = value
    elif isinstance(value, str):
        supported = "partial"
    else:
        supported = bool(value)
    return {
        "known": True,
        "supported": supported,
        "maximum_input_images": maximum,
        "supported_task": metadata.get("supported_multi_image_task"),
        "description": value if isinstance(value, str) else None,
        "source": str(source.relative_to(model_root)),
    }


def supports_multi_entry(
    capability: Mapping[str, Any], task: str, input_count: int,
) -> Optional[bool]:
    if input_count <= 1:
        return True
    if not capability.get("known"):
        return None
    supported = capability.get("supported")
    if supported is False:
        return False
    maximum = capability.get("maximum_input_images")
    if isinstance(maximum, int) and input_count > maximum:
        return False
    if supported == "partial":
        allowed_task = capability.get("supported_task")
        return bool(allowed_task and task == allowed_task)
    return bool(supported)


def result_is_current(path: Path, question_hash: str, judge_model: str) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    try:
        seed_is_current = int(record.get("seed", -1)) == 0
    except (TypeError, ValueError):
        seed_is_current = False
    return (
        record.get("protocol_version") == EVAL_PROTOCOL_VERSION
        and record.get("judge", {}).get("model") == judge_model
        and record.get("status") in {"scored", "unsupported_capability"}
        and record.get("question_record_sha256") == question_hash
        and record.get("judge", {}).get("prompt_templates_hash") == PROMPT_TEMPLATES_HASH
        and seed_is_current
    )


def write_json_atomic(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def evaluate_edit(
    judge: Judge,
    question_record: Mapping[str, Any],
    input_imgs: List,
    edited_img: Any,
) -> List[Dict[str, Any]]:
    input_count = len(input_imgs)
    if str(question_record.get("task", "")).startswith("ref/r1/"):
        image_description = (
            "Two images are attached. Image 1 is the reference image; "
            "Image 2 is the generated result. There is no separate base image."
        )
    elif input_count == 1:
        image_description = (
            "Two images are attached. Image 1 is the base input image; "
            "Image 2 is the edited result."
        )
    else:
        references = ", ".join(str(index) for index in range(2, input_count + 1))
        image_description = (
            f"{input_count + 1} images are attached. Image 1 is the base input image; "
            f"Images {references} are ordered reference images; "
            f"Image {input_count + 1} is the edited result."
        )

    results: List[Dict[str, Any]] = []
    for question in question_record["questions"]:
        user = USER_EDIT_TEMPLATE.format(
            image_description=image_description,
            instruction=question_record["instruction"],
            question=question["question"],
            success_condition=question["success_condition"],
            partial_condition=question["partial_condition"],
            failure_condition=question["failure_condition"],
        )
        score, raw = judge.answer(SYSTEM_EDIT, user, [*input_imgs, edited_img])
        results.append({
            "question_id": question["question_id"],
            "criterion": question["criterion"],
            "question": question["question"],
            "question_zh": question["question_zh"],
            "grounding": question.get("grounding", {}),
            "success_condition": question["success_condition"],
            "partial_condition": question["partial_condition"],
            "failure_condition": question["failure_condition"],
            "score": score,
            "raw": raw,
        })
    return results


def evaluate_gen(
    judge: Judge,
    question_record: Mapping[str, Any],
    generated_img: Any,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for question in question_record["questions"]:
        user = USER_GEN_TEMPLATE.format(
            prompt=question_record["instruction"],
            question=question["question"],
            success_condition=question["success_condition"],
            partial_condition=question["partial_condition"],
            failure_condition=question["failure_condition"],
        )
        score, raw = judge.answer(SYSTEM_GEN, user, [generated_img])
        results.append({
            "question_id": question["question_id"],
            "criterion": question["criterion"],
            "question": question["question"],
            "question_zh": question["question_zh"],
            "grounding": question.get("grounding", {}),
            "success_condition": question["success_condition"],
            "partial_condition": question["partial_condition"],
            "failure_condition": question["failure_condition"],
            "score": score,
            "raw": raw,
        })
    return results


def build_result_record(
    *,
    entry_id: str,
    item: Mapping[str, Any],
    question_record: Mapping[str, Any],
    question_hash: str,
    input_images: List[str],
    results: List[Dict[str, Any]],
    status: str,
    judge_info: Mapping[str, Any],
    seed_image: Optional[str],
    skip_reason: Optional[str] = None,
) -> Dict[str, Any]:
    scores = aggregate_question_scores(results, question_record["group_weights"])
    record: Dict[str, Any] = {
        "protocol_version": EVAL_PROTOCOL_VERSION,
        "status": status,
        "entry_id": entry_id,
        "sample_id": Path(entry_id).stem,
        "task": item.get("task", ""),
        "seed": 0,
        "kind": "gen" if str(item.get("task", "")).startswith("gen/") else "edit",
        "instruction": question_record["instruction"],
        "instruction_zh": question_record.get("instruction_zh"),
        "active_criteria": question_record.get("active_criteria", []),
        "group_weights": question_record["group_weights"],
        "questions": results,
        "scores": scores,
        "judge": dict(judge_info),
        "input_images": input_images,
        "seed_image": seed_image,
        "question_record_sha256": question_hash,
    }
    if skip_reason:
        record["skip_reason"] = skip_reason
    return record


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score seed_0 with the approved sample-specific MoI-Eval questions."
    )
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--judge", required=True,
                    choices=["openai"])
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-tag", default=None)
    ap.add_argument("--output-root", default=str(REPO_ROOT / "output"))
    ap.add_argument("--eval-json", default=str(DEFAULT_EVAL_JSON))
    ap.add_argument("--questions-jsonl", default=str(DEFAULT_QUESTIONS_JSONL))
    ap.add_argument("--mode", choices=["all", "gen", "edit"], default="all")
    ap.add_argument("--task", default=None)
    ap.add_argument("--entry-id", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="Compatibility option; only '--seeds 0' is accepted.")
    ap.add_argument("--multi-image-support", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of samples evaluated concurrently for remote judges "
            "(default: 1). Questions within one sample remain sequential."
        ),
    )
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if sorted(set(args.seeds)) != [0]:
        ap.error("MoI-Eval evaluates exactly one output per sample: --seeds 0")
    for tag in (args.model_tag, args.judge_tag or Path(args.judge_model).name):
        if not tag or Path(tag).name != tag or tag in {".", ".."}:
            ap.error("model and judge tags must be single directory names")
    if args.limit < 0:
        ap.error("--limit must be non-negative")
    if args.num_shards < 1:
        ap.error("--num-shards must be positive")
    if args.workers < 1:
        ap.error("--workers must be positive")
    env_world_size = int(os.environ.get("WORLD_SIZE", "0") or 0)
    env_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
    if args.num_shards == 1 and env_world_size > 1:
        args.num_shards = env_world_size
        args.shard_index = env_rank
    if not 0 <= args.shard_index < args.num_shards:
        ap.error("--shard-index must be in [0, --num-shards)")

    dataset = load_dataset(args.eval_json)
    items = dataset.items
    question_path = Path(args.questions_jsonl).expanduser().resolve()
    question_records = load_questions(question_path)
    model_root = Path(args.output_root).expanduser().resolve() / args.model_tag
    if not model_root.is_dir():
        print(f"[eval] model outputs dir not found: {model_root}", file=sys.stderr)
        return 1

    missing_questions = sorted(set(items) - set(question_records))
    extra_questions = sorted(set(question_records) - set(items))
    if missing_questions or extra_questions:
        raise ValueError(
            f"question/manifest mismatch: missing={len(missing_questions)} extra={len(extra_questions)}"
        )

    default_judge_tag = Path(args.judge_model).name.replace("/", "_")
    judge_tag = args.judge_tag or default_judge_tag
    eval_root = model_root / "eval" / judge_tag
    capability = read_multi_image_capability(model_root, args.multi_image_support)
    if not capability["known"]:
        print(
            "[eval] warning: multi-image capability is absent from _meta; "
            "use --multi-image-support yes/no if needed",
            file=sys.stderr,
        )

    jobs: List[Dict[str, Any]] = []
    for entry_id, item in items.items():
        sample_id = Path(entry_id).stem
        question_record = question_records[entry_id]
        task = str(item.get("task", ""))
        kind = "gen" if task.startswith("gen/") else "edit"
        if args.mode != "all" and kind != args.mode:
            continue
        if args.task and task != args.task:
            continue
        if args.entry_id and entry_id != args.entry_id:
            continue
        if question_record.get("task") != task:
            raise ValueError(f"{sample_id}: question task differs from manifest")
        if question_record.get("instruction") != item.get("instruction"):
            raise ValueError(f"{sample_id}: question instruction differs from manifest")

        sample_root = model_root / Path(*task.split("/")) / sample_id
        prompt_path = sample_root / "prompt.json"
        seed_path = sample_root / "seed_0.png"
        out_path = eval_root / Path(*task.split("/")) / sample_id / "seed_0.json"
        inputs = input_image_paths(entry_id, item)
        question_hash = stable_hash(question_record)
        multi_support = supports_multi_entry(capability, task, len(inputs))

        if kind == "edit" and len(inputs) > 1 and multi_support is False:
            job_type = "unsupported"
        elif not prompt_path.is_file() or not seed_path.is_file():
            continue
        else:
            try:
                metadata = json.loads(prompt_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as exc:
                print(f"[eval] skip invalid metadata {prompt_path}: {exc}", file=sys.stderr)
                continue
            if metadata.get("entry_id") != entry_id:
                print(f"[eval] skip {entry_id}: prompt.json entry_id differs", file=sys.stderr)
                continue
            if metadata.get("input_images", inputs) != inputs:
                print(f"[eval] skip {entry_id}: prompt.json input order differs", file=sys.stderr)
                continue
            job_type = "score"

        if not args.force and result_is_current(out_path, question_hash, args.judge_model):
            continue
        jobs.append({
            "type": job_type,
            "entry_id": entry_id,
            "sample_id": sample_id,
            "item": item,
            "questions": question_record,
            "question_hash": question_hash,
            "kind": kind,
            "input_images": inputs,
            "seed_path": seed_path,
            "out_path": out_path,
        })

    if args.limit:
        jobs = jobs[:args.limit]
    if args.num_shards > 1:
        jobs = [job for index, job in enumerate(jobs)
                if index % args.num_shards == args.shard_index]

    unsupported_count = sum(job["type"] == "unsupported" for job in jobs)
    print(
        f"[eval] model={args.model_tag} judge={judge_tag} pending={len(jobs)} "
        f"unsupported_zero={unsupported_count} seed_policy=seed_0_only "
        f"workers={args.workers} shard={args.shard_index}/{args.num_shards}"
    )
    if args.dry_run:
        for job in jobs[:10]:
            print(
                f"  - {job['type']} {job['kind']} {job['entry_id']} "
                f"({len(job['questions']['questions'])} questions) -> {job['out_path']}"
            )
        print(f"[eval] dry run: {len(jobs)} jobs")
        return 0

    eval_root.mkdir(parents=True, exist_ok=True)
    question_file_hash = hashlib.sha256(question_path.read_bytes()).hexdigest()
    if args.shard_index == 0:
        write_json_atomic(eval_root / "_judge_meta.json", {
            "protocol_version": EVAL_PROTOCOL_VERSION,
            "model_tag": args.model_tag,
            "judge_provider": args.judge,
            "judge_model": args.judge_model,
            "judge_tag": judge_tag,
            "questions_jsonl": str(question_path),
            "questions_file_sha256": question_file_hash,
            "prompt_templates_hash": PROMPT_TEMPLATES_HASH,
            "seed_policy": "seed_0_only",
            "workers": args.workers,
            "multi_image_capability": capability,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    judge: Optional[Judge] = None
    if any(job["type"] == "score" for job in jobs):
        judge = OpenAIJudge(model=args.judge_model)

    judge_info = {
        "provider": args.judge,
        "model": args.judge_model,
        "prompt_templates_hash": PROMPT_TEMPLATES_HASH,
    }

    def evaluate_job(job: Mapping[str, Any]) -> Dict[str, Any]:
        if job["type"] == "unsupported":
            reason = "model does not support this multi-image editing input"
            results = zero_question_results(job["questions"], reason)
            return build_result_record(
                entry_id=job["entry_id"], item=job["item"],
                question_record=job["questions"], question_hash=job["question_hash"],
                input_images=job["input_images"], results=results,
                status="unsupported_capability", judge_info=judge_info,
                seed_image=None, skip_reason=reason,
            )

        assert judge is not None
        from PIL import Image

        def load_rgb(path: Path) -> Any:
            with Image.open(path) as image:
                return image.convert("RGB")

        loaded_images = []
        try:
            output_image = load_rgb(job["seed_path"])
            loaded_images.append(output_image)
            if job["kind"] == "edit":
                source_images = []
                for relative_path in job["input_images"]:
                    source_path = resolve_data_path(dataset, relative_path)
                    if not source_path.is_file():
                        raise FileNotFoundError(f"missing input image: {source_path}")
                    source_image = load_rgb(source_path)
                    source_images.append(source_image)
                    loaded_images.append(source_image)
                results = evaluate_edit(
                    judge, job["questions"], source_images, output_image,
                )
            else:
                results = evaluate_gen(judge, job["questions"], output_image)
        finally:
            for image in loaded_images:
                image.close()

        return build_result_record(
            entry_id=job["entry_id"], item=job["item"],
            question_record=job["questions"], question_hash=job["question_hash"],
            input_images=job["input_images"], results=results,
            status="scored", judge_info=judge_info,
            seed_image=str(job["seed_path"].relative_to(model_root)),
        )

    ok = fail = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_jobs = {executor.submit(evaluate_job, job): job for job in jobs}
        for index, future in enumerate(as_completed(future_jobs), 1):
            job = future_jobs[future]
            try:
                record = future.result()
                write_json_atomic(job["out_path"], record)
                print(
                    f"[{index}/{len(jobs)}] {job['type']} {job['entry_id']} "
                    f"Q={record['scores']['usable_score']:.4f} "
                    f"t={time.time()-started:.1f}s"
                )
                ok += 1
            except Exception as exc:
                fail += 1
                print(
                    f"[{index}/{len(jobs)}] {job['entry_id']} FAIL: {exc}",
                    file=sys.stderr,
                )

    print(f"[eval] finished: ok={ok} fail={fail} in {time.time()-started:.1f}s")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
