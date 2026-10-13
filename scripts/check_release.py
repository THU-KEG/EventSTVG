"""Audit GitHub payload and optionally export a video-free ZIP, locally only."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".cache", ".venv", "venv",
                 "artifacts", "runtime", "outputs", "data", "evaluation_gt", "models",
                 "source_videos", "source_annotations", "data_build", "frozen_manifests", ".runs"}
SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{30,}|sk-(?:or-v1-)?[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})")


def payload():
    files, issues = [], []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if any(p in EXCLUDED_DIRS for p in rel.parts):
            continue
        if path.is_symlink():
            issues.append(f"symlink: {rel}")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".env", ".pem", ".key", ".pt", ".pth", ".safetensors", ".log"}:
            issues.append(f"forbidden payload type: {rel}")
            continue
        if path.stat().st_size > 5 * 1024**2:
            issues.append(f"file larger than 5 MiB: {rel}")
            continue
        try:
            text = path.read_text(encoding="utf8")
        except UnicodeError:
            issues.append(f"unexpected binary file: {rel}")
            continue
        if SECRET.search(text):
            issues.append(f"possible credential: {rel}")
        if re.search(r"/(?:home|data)/[A-Za-z0-9_-]+/", text):
            issues.append(f"workstation absolute path: {rel}")
        files.append(path)
    if issues:
        raise ValueError("Release audit failed:\n" + "\n".join(issues))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="Output path outside this repository")
    args = parser.parse_args()
    files = payload()
    if args.zip:
        if args.zip.resolve().is_relative_to(ROOT):
            parser.error("ZIP must be outside the repository")
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as output:
            for path in files:
                output.write(path, "EventSTVG/" + path.relative_to(ROOT).as_posix())
    print(json.dumps({"passed": True, "files": len(files),
                      "total_bytes": sum(p.stat().st_size for p in files),
                      "source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in files}}, indent=2))


if __name__ == "__main__":
    main()
