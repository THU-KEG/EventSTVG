"""Download only requested metadata or explicitly selected videos from HF."""
import argparse
import json
import os
from pathlib import Path

SETTINGS = ("oracle", "context_1min", "context_3min", "context_5min", "context_10min")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Use the confirmed HF dataset ID")
    parser.add_argument("--setting", choices=SETTINGS, required=True)
    parser.add_argument("--local-dir", type=Path, default=Path("data"))
    parser.add_argument("--revision", default="main", help="Pin a dataset revision for reproducibility")
    parser.add_argument("--sample-id", action="append", default=[], help="Explicitly request video for an ID; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Print metadata download plan without network")
    parser.add_argument("--proxy", default=os.environ.get("EVENTSTVG_PROXY"), help="Optional HTTP proxy; otherwise use normal environment proxy settings")
    args = parser.parse_args()
    folder = f"soccerstvg/videos/{args.setting}"
    if args.dry_run:
        print(json.dumps({"repo_id": args.repo_id, "revision": args.revision,
                          "metadata": f"{folder}/metadata.jsonl", "video_sample_ids": args.sample_id}))
        return
    if args.proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ[key] = args.proxy
        for key in ("NO_PROXY", "no_proxy"):
            os.environ[key] = ""
    from huggingface_hub import hf_hub_download
    def download(filename):
        return hf_hub_download(args.repo_id, filename, repo_type="dataset",
                               revision=args.revision, local_dir=args.local_dir)
    metadata = Path(download(f"{folder}/metadata.jsonl"))
    requested = set(args.sample_id)
    found = {}
    if requested:
        for line in metadata.open(encoding="utf8"):
            row = json.loads(line)
            if row["sample_id"] in requested:
                found[row["sample_id"]] = row["file_name"]
        missing = requested - found.keys()
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)}")
        for sid in args.sample_id:
            filename = Path(found[sid])
            if filename.is_absolute() or ".." in filename.parts:
                raise ValueError(f"Unsafe filename: {filename}")
            print(download(f"{folder}/{filename.as_posix()}"))
    print(metadata)


if __name__ == "__main__":
    main()
