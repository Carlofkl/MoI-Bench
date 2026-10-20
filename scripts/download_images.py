#!/usr/bin/env python3
"""Download and reconstruct the images used by MoI-Bench.

Pexels images are fetched through the official API.  Pass an API key with
``--pexels-api-key`` or the ``PEXELS_API_KEY`` environment variable.  The four
Wikimedia Commons images do not require a key.

The downloader treats these files as the source of truth:

* ``data/images/sources.csv``: source and license metadata;
* ``data/images/manifest.csv``: expected paths, dimensions and hashes;
* ``data/images/derivations.json``: deterministic normalization and
  synthetic corruption steps.

Images are written to exactly the relative paths used by samples.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import math
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
USER_AGENT = (
    "MoI-Bench/1.0 "
    "(https://github.com/Carlofkl/MoI-Bench; academic research)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pexels-api-key",
        default=os.environ.get("PEXELS_API_KEY", ""),
        help="Pexels API key (or set PEXELS_API_KEY).",
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Directory containing samples.json and images metadata.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        help="Write paths relative to this directory (default: --data-root).",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        help="Raw-download cache (default: DATA_ROOT/.download_cache).",
    )
    parser.add_argument(
        "--only", action="append", default=[], metavar="TEXT",
        help="Only process paths containing TEXT; may be repeated.",
    )
    parser.add_argument("--limit", type=int, help="Process at most N selected rows.")
    parser.add_argument("--force", action="store_true", help="Overwrite non-matching files.")
    parser.add_argument(
        "--refresh-cache", action="store_true",
        help="Download provider files again instead of using the raw cache.",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Check local files without making network requests.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions only.")
    parser.add_argument(
        "--strict-hash", action="store_true",
        help="Exit non-zero if reconstructed bytes differ from the manifest hash.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_destination(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe manifest path: {relative!r}")
    destination = (root / rel).resolve()
    resolved_root = root.resolve()
    if destination != resolved_root and resolved_root not in destination.parents:
        raise ValueError(f"Path escapes output root: {relative!r}")
    return destination


def request_bytes(url: str, headers: dict[str, str] | None = None,
                  attempts: int = 5) -> bytes:
    request_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    request_headers.update(headers or {})
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt + 1 == attempts:
                raise
            delay = float(exc.headers.get("Retry-After") or min(30, 2 ** (attempt + 1)))
            print(f"  HTTP {exc.code}; retry in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
        except (
            TimeoutError,
            urllib.error.URLError,
            ConnectionError,
            http.client.HTTPException,
        ):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(20, 2 ** (attempt + 1)))
    raise RuntimeError(f"Unable to download {url}")


def pexels_image_url(source_id: str, variant: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError(
            "Pexels images require --pexels-api-key or PEXELS_API_KEY."
        )
    api_url = f"https://api.pexels.com/v1/photos/{urllib.parse.quote(source_id)}"
    try:
        body = request_bytes(
            api_url,
            {"Authorization": api_key, "Accept": "application/json"},
        )
    except urllib.error.HTTPError as exc:
        # A small number of public Pexels pages/CDN assets are no longer
        # returned by the photo-detail API.  Pexels' canonical large2x URL is
        # still public and produces the same bytes as the API's large2x field.
        if exc.code == 404 and variant == "large2x" and source_id.isdigit():
            print(
                f"  Pexels API has no photo record for {source_id}; "
                "using its canonical large2x CDN URL.",
                file=sys.stderr,
            )
            return pexels_large2x_cdn_url(source_id)
        raise
    photo = json.loads(body.decode("utf-8"))
    sources = photo.get("src") or {}
    for key in (variant, "large2x", "original", "large"):
        if sources.get(key):
            return str(sources[key])
    raise RuntimeError(f"Pexels photo {source_id} has no downloadable URL")


def pexels_large2x_cdn_url(source_id: str) -> str:
    """Return the canonical URL used by Pexels for its ``large2x`` variant."""
    quoted_id = urllib.parse.quote(source_id)
    return (
        f"https://images.pexels.com/photos/{quoted_id}/"
        f"pexels-photo-{quoted_id}.jpeg"
        "?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940"
    )


def commons_image_url(source_url: str) -> str:
    parsed = urllib.parse.urlparse(source_url)
    title = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
    filename = title.removeprefix("File:")
    if not filename:
        raise RuntimeError(f"Could not resolve Wikimedia source: {source_url}")
    return (
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
        + urllib.parse.quote(filename, safe="")
    )


def raw_cache_path(cache_dir: Path, provider: str, source_id: str) -> Path:
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_id)
    return cache_dir / provider / f"{safe_id}.source"


def get_raw_source(row: dict[str, str], license_row: dict[str, str],
                   cache_dir: Path, api_key: str, refresh_cache: bool = False) -> bytes:
    provider = row["provider"].strip().lower()
    source_id = row["source_id"].strip()
    cache_path = raw_cache_path(cache_dir, provider, source_id)
    if cache_path.exists() and not refresh_cache:
        return cache_path.read_bytes()
    if provider == "pexels":
        url = pexels_image_url(source_id, row.get("download_variant", "large2x"), api_key)
    elif provider in {"wikimedia", "wikimedia_commons", "commons"}:
        url = commons_image_url(license_row["source_url"])
    else:
        raise RuntimeError(f"Unsupported provider {provider!r} for {row['local_path']}")
    content = request_bytes(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)
    return content


def resampling(name: str):
    from PIL import Image

    namespace = getattr(Image, "Resampling", Image)
    return getattr(namespace, name)


def normalized_image(raw: bytes, size: tuple[int, int], resize_filter: str = "LANCZOS"):
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if image.size != size:
        image = image.resize(size, resampling(resize_filter))
    return image


def overlay_font(size: int):
    from PIL import ImageFont

    candidates = (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def apply_operation(image, operation: dict[str, Any]):
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    name = operation["operation"]
    if name == "add_gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(float(operation.get("radius", 2.4))))
    if name in {"add_sensor_noise", "add_dim_noise"}:
        rng = random.Random(int(operation["seed"]))
        noise = Image.frombytes("L", image.size, rng.randbytes(image.width * image.height))
        noise_rgb = Image.merge("RGB", (noise, noise, noise))
        if name == "add_dim_noise":
            image = ImageEnhance.Brightness(image).enhance(0.48)
        return Image.blend(image, noise_rgb, float(operation["noise_opacity"]))
    if name == "add_motion_blur":
        # Average translated copies of the image.  Integer offsets and fixed
        # parameters keep the degradation deterministic across machines.
        length = max(3, int(operation.get("length", 13)))
        if length % 2 == 0:
            length += 1
        angle = math.radians(float(operation.get("angle_degrees", 0.0)))
        radius = length // 2
        frames = []
        for offset in range(-radius, radius + 1):
            dx = int(round(math.cos(angle) * offset))
            dy = int(round(math.sin(angle) * offset))
            shifted = image.copy()
            shifted.paste(image, (dx, dy))
            frames.append(shifted)
        blurred = frames[0]
        for count, frame in enumerate(frames[1:], start=2):
            blurred = Image.blend(blurred, frame, 1.0 / count)
        return blurred
    if name == "add_moire_pattern":
        # Wavy, closely spaced chromatic lines approximate the interference
        # seen when a display or fine repeated texture is photographed.
        spacing = max(4, int(operation.get("spacing", 7)))
        amplitude = float(operation.get("wave_amplitude", 2.4))
        wavelength = max(24.0, float(operation.get("wave_period", 92.0)))
        opacity = max(0.0, min(1.0, float(operation.get("opacity", 0.16))))
        angle = float(operation.get("angle_degrees", 0.0))
        phase = float(operation.get("phase", 0.0))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        alpha = max(1, int(round(255 * opacity)))
        colors = ((35, 165, 225, alpha), (225, 45, 155, alpha), (245, 185, 45, alpha))
        step = max(3, image.width // 300)
        margin = int(abs(amplitude)) + 4
        for line_index, base_y in enumerate(range(-margin, image.height + margin, spacing)):
            points = []
            for x in range(0, image.width + step, step):
                y = base_y + amplitude * math.sin((2.0 * math.pi * x / wavelength) + phase)
                points.append((x, y))
            draw.line(points, fill=colors[line_index % len(colors)], width=1)
        if angle:
            overlay = overlay.rotate(
                angle,
                resample=resampling("BICUBIC"),
                expand=False,
                fillcolor=(0, 0, 0, 0),
            )
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    if name == "add_low_resolution":
        scale = float(operation.get("downsample_scale", 0.28))
        small = (max(64, int(image.width * scale)), max(64, int(image.height * scale)))
        return image.resize(small, resampling("BILINEAR")).resize(
            image.size, resampling("BILINEAR")
        )
    if name in {"add_yellow_color_cast", "add_cyan_color_cast"}:
        color = tuple(operation["overlay_rgb"])
        return Image.blend(image, Image.new("RGB", image.size, color),
                           float(operation["opacity"]))
    if name == "add_underexposure":
        return ImageEnhance.Brightness(image).enhance(float(operation["brightness_factor"]))
    if name == "add_overexposure":
        image = ImageEnhance.Brightness(image).enhance(float(operation["brightness_factor"]))
        return ImageEnhance.Contrast(image).enhance(float(operation["contrast_factor"]))
    if name == "add_uneven_shadow":
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((-int(image.width * .15), int(image.height * .05),
                      int(image.width * .62), int(image.height * .78)), fill=175)
        mask = mask.filter(ImageFilter.GaussianBlur(max(18, min(image.size) // 18)))
        dark = ImageEnhance.Brightness(image).enhance(float(operation["brightness_factor"]))
        return Image.composite(dark, image, mask)
    if name == "add_haze":
        opacity = float(operation.get("haze_opacity", 0.3))
        return Image.blend(image, Image.new("RGB", image.size, (235, 239, 242)), opacity)
    if name == "add_faded_color":
        image = ImageEnhance.Color(image).enhance(float(operation["color_factor"]))
        return ImageEnhance.Contrast(image).enhance(float(operation["contrast_factor"]))
    if name == "add_jpeg_artifacts":
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=int(operation.get("jpeg_quality", 14)))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if name == "add_dust_and_scratches":
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        rng = random.Random(int(operation["seed"]))
        for _ in range(int(operation.get("dust_count", 18))):
            x, y = rng.randrange(image.width), rng.randrange(image.height)
            draw.ellipse((x, y, x + rng.randrange(2, 9), y + rng.randrange(2, 9)),
                         fill=(238, 232, 214, 170))
        for _ in range(int(operation.get("scratch_count", 7))):
            x = rng.randrange(image.width)
            length = rng.randrange(max(20, image.height // 12), max(21, image.height // 3))
            y = rng.randrange(max(1, image.height - length))
            draw.line((x, y, x + rng.randrange(-8, 9), y + length),
                      fill=(242, 236, 220, 145), width=max(1, image.width // 650))
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    if name == "add_visible_text_watermark":
        rgba = image.convert("RGBA")
        overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = overlay_font(max(30, min(image.size) // 11))
        # Preserve the historical pixel-reconstruction fallback, not project branding.
        text = str(operation.get("text", "MOGE REVIEW SAMPLE"))
        box = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = max(1, box[2] - box[0]), max(1, box[3] - box[1])
        x, y = max(12, (image.width - text_width) // 2), max(12, (image.height - text_height) // 2)
        draw.rounded_rectangle(
            (x - 18, y - 12, min(image.width - 1, x + text_width + 18),
             min(image.height - 1, y + text_height + 12)),
            radius=max(8, text_height // 4), fill=(255, 255, 255, 75),
        )
        draw.text((x, y), text, font=font, fill=(205, 25, 45, 150),
                  stroke_width=max(1, text_height // 28),
                  stroke_fill=(255, 255, 255, 120))
        return Image.alpha_composite(rgba, overlay).convert("RGB")
    if name == "add_visible_inpaint_mask":
        draw = ImageDraw.Draw(image, "RGBA")
        box = tuple(int(value) for value in operation["box_xyxy"])
        if operation.get("shape") == "ellipse":
            draw.ellipse(box, fill=(115, 125, 140, 235))
        else:
            draw.rounded_rectangle(box, radius=max(8, min(image.size) // 45),
                                   fill=(105, 115, 130, 235))
        return image
    if name == "legacy_colored_rectangles":
        draw = ImageDraw.Draw(image)
        for box in operation["boxes"]:
            draw.rectangle(tuple(box["xyxy"]), fill=tuple(box["rgb"]))
        return image
    if name == "legacy_large_occlusion":
        ImageDraw.Draw(image).polygon(
            [tuple(point) for point in operation["points"]],
            fill=tuple(operation.get("rgb", [120, 120, 120])),
        )
        return image
    raise RuntimeError(f"Unknown derivation operation: {name}")


def reconstruct(raw: bytes, manifest_row: dict[str, str],
                derivation: dict[str, Any] | None) -> bytes:
    from PIL import Image, ImageOps

    size = (int(manifest_row["width"]), int(manifest_row["height"]))
    mode = manifest_row.get("reconstruction_mode", "raw_large2x")
    if mode == "raw_large2x" and not derivation:
        with Image.open(io.BytesIO(raw)) as opened:
            observed = ImageOps.exif_transpose(opened).size
        if observed == size:
            return raw

    spec = derivation or {"target_size": list(size), "jpeg_quality": 92, "optimize": True,
                          "operations": []}
    resize_filter = "BICUBIC" if mode == "legacy_stress" else "LANCZOS"
    image = normalized_image(raw, tuple(spec.get("target_size", size)), resize_filter)
    operations = spec.get("operations") or []

    # Candidate images were normalized to JPEG and reopened before preprocessing.
    if operations and mode == "derived_jpeg":
        intermediate = io.BytesIO()
        image.save(intermediate, "JPEG", quality=92, optimize=True)
        intermediate.seek(0)
        image = Image.open(intermediate).convert("RGB")
    for operation in operations:
        image = apply_operation(image, operation)

    result = io.BytesIO()
    image.save(
        result, format="JPEG", quality=int(spec.get("jpeg_quality", 92)),
        optimize=bool(spec.get("optimize", True)),
    )
    return result.getvalue()


def validate_metadata(manifest: list[dict[str, str]], licenses: list[dict[str, str]],
                      derivations: dict[str, Any]) -> dict[str, dict[str, str]]:
    manifest_paths = [row["local_path"] for row in manifest]
    license_paths = [row["local_path"] for row in licenses]
    if len(set(manifest_paths)) != len(manifest_paths):
        raise RuntimeError("manifest.csv contains duplicate local_path values")
    if len(set(license_paths)) != len(license_paths):
        raise RuntimeError("sources.csv contains duplicate local_path values")
    if set(manifest_paths) != set(license_paths):
        missing_license = sorted(set(manifest_paths) - set(license_paths))
        missing_manifest = sorted(set(license_paths) - set(manifest_paths))
        raise RuntimeError(
            f"Metadata path mismatch: no license={missing_license[:3]}, "
            f"no manifest={missing_manifest[:3]}"
        )
    unknown = set(derivations) - set(manifest_paths)
    if unknown:
        raise RuntimeError(f"derivations.json has unknown paths: {sorted(unknown)[:3]}")
    return {row["local_path"]: row for row in licenses}


def select_rows(rows: Iterable[dict[str, str]], patterns: list[str],
                limit: int | None) -> list[dict[str, str]]:
    selected = [row for row in rows if not patterns or any(
        pattern in row["local_path"] for pattern in patterns
    )]
    return selected[:limit] if limit is not None else selected


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = (args.output_root or data_root).resolve()
    cache_dir = (args.cache_dir or data_root / ".download_cache").resolve()
    manifest = read_csv(data_root / "images" / "manifest.csv")
    licenses = read_csv(data_root / "images" / "sources.csv")
    derivation_doc = json.loads((data_root / "images" / "derivations.json").read_text(
        encoding="utf-8"
    ))
    derivations = derivation_doc.get("items") or {}
    license_by_path = validate_metadata(manifest, licenses, derivations)
    rows = select_rows(manifest, args.only, args.limit)

    if not args.verify_only and not args.dry_run and not args.pexels_api_key:
        for row in rows:
            if row["provider"].strip().lower() != "pexels":
                continue
            destination = safe_destination(output_root, row["local_path"])
            needs_download = not destination.exists()
            if destination.exists() and args.force:
                needs_download = sha256(destination) != row["sha256"]
            cache_path = raw_cache_path(
                cache_dir,
                row["provider"].strip().lower(),
                row["source_id"].strip(),
            )
            if needs_download and (args.refresh_cache or not cache_path.exists()):
                raise SystemExit(
                    "Pexels images are missing or need replacement. Pass "
                    "--pexels-api-key or set PEXELS_API_KEY."
                )

    counts = {"planned": 0, "ok": 0, "downloaded": 0, "missing": 0, "mismatch": 0, "failed": 0}
    print(f"MoI-Bench images: selected {len(rows)} / {len(manifest)}")
    for index, row in enumerate(rows, 1):
        relative = row["local_path"]
        destination = safe_destination(output_root, relative)
        expected = row["sha256"]
        if destination.exists():
            observed = sha256(destination)
            if observed == expected:
                counts["ok"] += 1
                continue
            if args.verify_only or not args.force:
                counts["mismatch"] += 1
                print(f"[mismatch] {relative}: {observed[:12]} != {expected[:12]}")
                continue
        elif args.verify_only:
            counts["missing"] += 1
            print(f"[missing]  {relative}")
            continue

        action = "would download" if args.dry_run else "download"
        print(f"[{index:04d}/{len(rows):04d}] {action} {relative}")
        if args.dry_run:
            counts["planned"] += 1
            continue
        try:
            raw = get_raw_source(
                row, license_by_path[relative], cache_dir, args.pexels_api_key,
                refresh_cache=args.refresh_cache,
            )
            content = reconstruct(raw, row, derivations.get(relative))
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(content)
            temporary.replace(destination)
            observed = hashlib.sha256(content).hexdigest()
            if observed != expected:
                counts["mismatch"] += 1
                print(
                    f"  [hash warning] {relative}: reconstructed {observed[:12]}, "
                    f"manifest {expected[:12]}", file=sys.stderr,
                )
            counts["downloaded"] += 1
        except Exception as exc:  # report individual failures and continue the batch
            counts["failed"] += 1
            print(f"  [failed] {relative}: {exc}", file=sys.stderr)

    print("Summary: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    if counts["failed"] or (args.strict_hash and (counts["mismatch"] or counts["missing"])):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
