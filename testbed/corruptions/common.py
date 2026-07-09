"""Shared infrastructure for the corruption testbed.

Every corruption is a pure function: (PIL.Image, severity, rng) -> (PIL.Image, params dict).
Every output image carries a full construction record so its label is
reconstructible fact, never opinion (RIGOR_PLAYBOOK #1).

Severity convention: 1 = subtle-but-real, 2 = clear, 3 = unmistakable.
"""

import hashlib
import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SPLITS = REPO / "data" / "features" / "splits_v2.json"
RAW = REPO / "data" / "scrape" / "raw"
OUT_DIR = REPO / "data" / "testset" / "corruptions"

GENERATOR_VERSION = "0.1.0"
SEVERITIES = (1, 2, 3)
MAX_SIDE = 1024  # normalize so corruption magnitudes mean the same thing everywhere


def image_path(image_id: str) -> Path:
    """'fb:rhode/x.jpg' -> data/scrape/raw/images/rhode/x.jpg ; 'ig:...' -> ig_images."""
    src, rel = image_id.split(":", 1)
    folder = "images" if src == "fb" else "ig_images"
    return RAW / folder / rel


def load_base(image_id: str) -> Image.Image:
    img = Image.open(image_path(image_id)).convert("RGB")
    if max(img.size) > MAX_SIDE:
        scale = MAX_SIDE / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.LANCZOS)
    return img


def rhode_test_ids(split: str = "test") -> list[str]:
    """All rhode image ids in the given split (default: the random-cluster test split).

    The temporal holdout is NOT used for corruption bases — it stays pure for
    its own axis.
    """
    data = json.loads(SPLITS.read_text())
    index = data["_image_index"]
    return sorted(i for i, sp in index.items()
                  if sp == split and i.split(":")[1].split("/")[0] == "rhode")


def sample_bases(n: int, seed: int, split: str = "test") -> list[str]:
    ids = rhode_test_ids(split)
    rng = random.Random(seed)
    return rng.sample(ids, min(n, len(ids)))


@dataclass
class CorruptionRecord:
    """The construction certificate for one corrupted image."""
    out_file: str            # relative to OUT_DIR
    source_id: str           # corpus image id of the clean base
    dimension: str           # palette | composition | typography
    corruption: str          # generator name, e.g. 'hue_rotation'
    severity: int            # 1..3
    params: dict             # exact parameters used — severity IS these numbers
    seed: int                # per-image RNG seed
    generator_version: str = GENERATOR_VERSION
    label: str = "off-brand"
    sha1: str = ""           # of the written file, for lineage checks
    notes: dict = field(default_factory=dict)


def out_name(rec_source_id: str, corruption: str, severity: int) -> str:
    src_slug = rec_source_id.replace(":", "_").replace("/", "_")
    return f"{corruption}/s{severity}/{src_slug}.jpg"


def write_output(img: Image.Image, rec: CorruptionRecord) -> CorruptionRecord:
    """JPEG q95 no-subsampling: positives are JPEGs, so corruptions must be too —
    a PNG-vs-JPEG format split would hand judges a compression shortcut."""
    path = OUT_DIR / rec.out_file
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=95, subsampling=0)
    rec.sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
    return rec


def append_manifest(records: list[CorruptionRecord], manifest_name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / manifest_name
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    return path


def contact_sheet(rows: list[list[Image.Image]], labels: list[str],
                  cell: int = 256) -> Image.Image:
    """Grid: one row per base image [clean, s1, s2, s3], header labels per column."""
    from PIL import ImageDraw
    ncols = max(len(r) for r in rows)
    header = 28
    sheet = Image.new("RGB", (ncols * cell, header + len(rows) * cell), "white")
    draw = ImageDraw.Draw(sheet)
    for c, lab in enumerate(labels[:ncols]):
        draw.text((c * cell + 8, 6), lab, fill="black")
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            thumb = im.copy()
            thumb.thumbnail((cell, cell))
            sheet.paste(thumb, (c * cell + (cell - thumb.width) // 2,
                                header + r * cell + (cell - thumb.height) // 2))
    return sheet
