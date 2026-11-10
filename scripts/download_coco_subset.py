"""Download official COCO 2014 captions and a reproducible validation subset."""
import argparse
import json
import random
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests
from PIL import Image

from palak_repro.common import CLASSES

# The COCO project's own download page publishes these HTTP endpoints.
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
IMAGE_URL = "http://images.cocodataset.org/val2014/{}"


def download(url, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with requests.get(url, stream=True, timeout=(15, 30)) as response:
        response.raise_for_status()
        with temporary.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)
    temporary.replace(output)


def select(annotation_path, count, seed):
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    names = {x["id"]: x["file_name"] for x in annotations["images"]}
    records, seen = [], set()
    for entry in annotations["annotations"]:
        image_id = entry["image_id"]
        caption = " ".join(entry["caption"].split())
        if image_id in seen or not caption:
            continue
        if any(re.search(rf"\b{re.escape(c)}\b", caption, flags=re.IGNORECASE) for c in CLASSES):
            continue
        records.append({"image_id": image_id, "file_name": names[image_id], "caption": caption})
        seen.add(image_id)
    if len(records) < count:
        raise ValueError(f"Only {len(records)} eligible COCO images; requested {count}")
    return random.Random(seed).sample(records, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    annotation = args.root / "annotations" / "captions_val2014.json"
    if not annotation.is_file():
        with tempfile.TemporaryDirectory(prefix="coco-annotations-") as directory:
            archive = Path(directory) / "annotations_trainval2014.zip"
            download(ANNOTATIONS_URL, archive)
            with zipfile.ZipFile(archive) as source:
                member = "annotations/captions_val2014.json"
                if source.testzip() is not None:
                    raise ValueError("COCO annotation archive failed CRC verification")
                annotation.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_file, annotation.open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file)
    records = select(annotation, args.count, args.seed)
    images = args.root / "val2014"
    for index, record in enumerate(records, 1):
        path = images / record["file_name"]
        if not path.is_file():
            download(IMAGE_URL.format(record["file_name"]), path)
        with Image.open(path) as image:
            image.verify()
        if index % 10 == 0 or index == len(records):
            print(f"Verified {index}/{len(records)} COCO reference images", flush=True)
    (args.root / "coco_subset.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
