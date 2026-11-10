"""Deterministic COCO 2014 caption and paired reference image sampling."""
import json
import random
import re
from pathlib import Path

from .common import CLASSES


def coco_samples(annotation_path, images_dir, count, seed, exclude_classes=True):
    if count == 0:
        return []
    annotations = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    names = {x["id"]: x["file_name"] for x in annotations["images"]}
    records = []
    seen = set()
    for entry in annotations["annotations"]:
        image_id = entry["image_id"]
        caption = " ".join(entry["caption"].split())
        if image_id in seen or not caption:
            continue
        if exclude_classes and any(re.search(rf"\b{re.escape(c)}\b", caption, flags=re.IGNORECASE) for c in CLASSES):
            continue
        path = Path(images_dir) / names[image_id]
        if path.exists():
            records.append({"image_id": image_id, "caption": caption, "reference": str(path)})
            seen.add(image_id)
    if len(records) < count:
        raise ValueError(f"Only {len(records)} valid COCO image/caption pairs; requested {count}")
    return random.Random(seed).sample(records, count)
