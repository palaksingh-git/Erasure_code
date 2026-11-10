"""Download and extract only Phan's pretrained CIFAR-10 ResNet-18 weights."""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import gdown

GOOGLE_DRIVE_ID = "17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq"


def extract(archive, output):
    with zipfile.ZipFile(archive) as zipped:
        names = [n for n in zipped.namelist() if n.endswith("/resnet18.pt") or n == "resnet18.pt"]
        if len(names) != 1:
            raise ValueError(f"Expected one resnet18.pt in archive, found {names}")
        with zipped.open(names[0]) as source, output.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    print(f"Saved {output} ({output.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/classifier/resnet18.pt")
    parser.add_argument("--archive", help="Use an already downloaded Phan weights archive")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.archive:
        archive = Path(args.archive)
        if not archive.is_file():
            raise FileNotFoundError(archive)
        extract(archive, output)
        return
    # gdown opens its output for writing; a live NamedTemporaryFile is locked on Windows.
    with tempfile.TemporaryDirectory(prefix="phan-cifar-") as directory:
        archive = Path(directory) / "weights.zip"
        downloaded = gdown.download(id=GOOGLE_DRIVE_ID, output=str(archive), quiet=False)
        if not downloaded:
            raise RuntimeError("Phan classifier archive download failed; use --archive with a local copy")
        extract(archive, output)


if __name__ == "__main__":
    main()
