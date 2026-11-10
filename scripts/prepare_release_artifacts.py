"""Prepare a portable manifest for the project's local data and checkpoints."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

PART_BYTES = 1_800_000_000  # below GitHub's 2 GiB release-asset limit


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def archive_directory(root, output, prefix=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                         allowZip64=True) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                name = (Path(prefix) / path.relative_to(root)).as_posix() if prefix else path.relative_to(root).as_posix()
                archive.write(path, name)
    return {"asset": output.name, "size": output.stat().st_size, "sha256": digest(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--venv-source", type=Path)
    parser.add_argument("--source-git", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    upload = root / "results" / "migration_upload"
    upload.mkdir(parents=True, exist_ok=True)
    files = sorted(path for folder in ("checkpoints", "results", "data")
                   for path in (root / folder).rglob("*") if path.is_file()
                   and not path.is_relative_to(upload)
                   and path.name != "migration_auth_check.py"
                   and path.relative_to(root).as_posix() != "data/README.md")
    small, large = [], []
    unique = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if size < PART_BYTES:
            small.append({"path": relative, "size": size, "sha256": digest(path)})
            continue
        overall = hashlib.sha256()
        parts = []
        with path.open("rb") as stream:
            index = 0
            while True:
                part_hash = hashlib.sha256()
                remaining = PART_BYTES
                part_size = 0
                while remaining:
                    block = stream.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        break
                    overall.update(block)
                    part_hash.update(block)
                    remaining -= len(block)
                    part_size += len(block)
                if part_size == 0:
                    break
                index += 1
                parts.append({"index": index, "size": part_size, "sha256": part_hash.hexdigest()})
        checksum = overall.hexdigest()
        for part in parts:
            part["asset"] = f"checkpoint-{checksum[:20]}-part{part['index']:02d}.bin"
        unique[checksum] = {"path": relative, "size": size, "sha256": checksum, "parts": parts}
        large.append({"path": relative, "size": size, "sha256": checksum})
        print(f"Hashed {relative}: {checksum}", flush=True)
    small_zip = upload / "small-artifacts.zip"
    with zipfile.ZipFile(small_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                         allowZip64=True) as archive:
        for item in small:
            archive.write(root / item["path"], item["path"])
    manifest = {
        "schema_version": 1,
        "repository": "palaksingh-git/Erasure_code",
        "release_tag": "full-project-artifacts-2026-11-10",
        "small_archive": {"asset": small_zip.name, "size": small_zip.stat().st_size,
                          "sha256": digest(small_zip), "files": small},
        "large_files": large,
        "unique_checkpoints": sorted(unique.values(), key=lambda value: value["sha256"]),
    }
    if args.venv_source:
        manifest["optional_windows_venv"] = archive_directory(args.venv_source.resolve(),
                                                                upload / "windows-venv.zip", ".venv")
    if args.source_git:
        bundle = upload / "source-history.bundle"
        subprocess.run(["git", "-C", str(args.source_git), "bundle", "create", str(bundle), "--all"], check=True)
        manifest["source_history_bundle"] = {"asset": bundle.name, "size": bundle.stat().st_size,
                                              "sha256": digest(bundle)}
    output = root / "artifacts" / "release-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}; {len(small)} small files and {len(large)} checkpoint paths"
          f" ({len(unique)} unique payloads)", flush=True)


if __name__ == "__main__":
    main()
