"""Shared MoI benchmark loading and output-path helpers.

The benchmark JSON is the source of truth for both task membership and input
image order.  Keeping this logic here prevents individual model wrappers from
reconstructing paths from legacy file names or task labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_JSON = REPO_ROOT / "data" / "samples.json"


@dataclass(frozen=True)
class MoiDataset:
    eval_json: Path
    data_root: Path
    version: str
    document: Mapping[str, Any]
    items: Mapping[str, Mapping[str, Any]]


def load_dataset(eval_json: Path | str = DEFAULT_EVAL_JSON) -> MoiDataset:
    """Load a MoI manifest and resolve its declared relative-path base."""
    manifest = Path(eval_json).expanduser().resolve()
    document = json.loads(manifest.read_text(encoding="utf-8"))
    items = document.get("items")
    if not isinstance(items, dict):
        raise ValueError(f"{manifest}: expected an object-valued 'items' field")

    path_base = Path(str(document.get("path_base", ".")))
    if path_base.is_absolute():
        raise ValueError(f"{manifest}: path_base must be relative, got {path_base}")
    data_root = (manifest.parent / path_base).resolve()
    return MoiDataset(
        eval_json=manifest,
        data_root=data_root,
        version=str(document.get("schema_version", "unknown")),
        document=document,
        items=items,
    )


def is_generation(item: Mapping[str, Any]) -> bool:
    """Use the taxonomy, not the entry-id spelling, to identify generation."""
    return str(item.get("task", "")).split("/", 1)[0] == "gen"


def input_image_paths(entry_id: str, item: Mapping[str, Any]) -> List[str]:
    """Return model inputs in benchmark order.

    Generation has no image input.  Every editing entry uses the image stored
    under its entry id as B1/R1; optional ``reference_images`` follow it in the
    exact order recorded by the manifest.  ``reference_image`` remains accepted
    for compatibility with an older one-reference manifest.
    """
    if is_generation(item):
        return []

    refs = item.get("reference_images")
    if refs is None:
        legacy_ref = item.get("reference_image")
        refs = [legacy_ref] if legacy_ref else []
    if not isinstance(refs, list) or not all(isinstance(path, str) for path in refs):
        raise ValueError(f"{entry_id}: reference_images must be a list of paths")
    return [entry_id, *refs]


def resolve_data_path(dataset: MoiDataset, relative_path: str) -> Path:
    """Resolve a manifest path while rejecting paths outside ``path_base``."""
    rel = Path(relative_path)
    if rel.is_absolute():
        raise ValueError(f"Image path must be relative: {relative_path}")
    resolved = (dataset.data_root / rel).resolve()
    try:
        resolved.relative_to(dataset.data_root)
    except ValueError as exc:
        raise ValueError(f"Image path escapes path_base: {relative_path}") from exc
    return resolved
