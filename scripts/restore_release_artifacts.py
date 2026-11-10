"""Restore large project artifacts from this repository's GitHub release."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def matches(path, item):
    return path.is_file() and path.stat().st_size == item["size"] and sha256(path) == item["sha256"]


def download(url, path, item):
    if matches(path, item):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    existing = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "Erasure-project-restorer"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        append = existing > 0 and response.status == 206
        with partial.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, 8 * 1024 * 1024)
    if not matches(partial, item):
        raise ValueError(f"Downloaded asset failed size or SHA-256 verification: {path.name}")
    partial.replace(path)


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes project root: {relative}")
    return path


def restore_zip(root, archive_path, files):
    expected = {item["path"]: item for item in files}
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError("Archive entries differ from the manifest")
        if archive.testzip() is not None:
            raise ValueError("Archive failed CRC verification")
        for relative, item in expected.items():
            target = safe_path(root, relative)
            if matches(target, item):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".restore.part")
            with archive.open(relative) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
            if not matches(partial, item):
                raise ValueError(f"Restored file failed SHA-256 verification: {relative}")
            partial.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--include-venv", action="store_true",
                        help="Also restore the original Windows virtual environment")
    parser.add_argument("--include-history", action="store_true",
                        help="Download the source Git bundle without replacing this repository's history")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "artifacts" / "release-manifest.json").read_text(encoding="utf-8"))
    url = (f"https://api.github.com/repos/{manifest['repository']}/releases/tags/"
           f"{manifest['release_tag']}")
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                  "User-Agent": "Erasure-project-restorer"})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    urls = {item["name"]: item["browser_download_url"] for item in release["assets"]}
    cache = root / "results" / "migration_download"

    def asset(item):
        name = item["asset"]
        if name not in urls:
            raise FileNotFoundError(f"Release asset missing: {name}")
        path = cache / name
        download(urls[name], path, item)
        return path

    small = manifest["small_archive"]
    restore_zip(root, asset(small), small["files"])
    unique = {item["sha256"]: item for item in manifest["unique_checkpoints"]}
    for item in manifest["large_files"]:
        target = safe_path(root, item["path"])
        if matches(target, item):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".restore.part")
        with partial.open("wb") as output:
            for part in unique[item["sha256"]]["parts"]:
                with asset(part).open("rb") as source:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
        if not matches(partial, item):
            raise ValueError(f"Checkpoint failed SHA-256 verification: {item['path']}")
        partial.replace(target)
        print(f"Restored {item['path']}", flush=True)
    if args.include_venv:
        archive = asset(manifest["optional_windows_venv"])
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("Windows virtual environment archive failed CRC verification")
            for member in zipped.namelist():
                safe_path(root, member)
            zipped.extractall(root)
    if args.include_history:
        asset(manifest["source_history_bundle"])
    print("Artifact restoration complete", flush=True)


if __name__ == "__main__":
    main()
