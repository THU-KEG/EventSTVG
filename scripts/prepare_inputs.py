"""Export model-input-only manifests from one integrated metadata JSONL."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gemini-yxyx", action="store_true", help="Historical Gemini coordinate prompt adaptation")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with args.metadata.open(encoding="utf8") as source, args.out.open("x", encoding="utf8") as output:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["sample_id"]
            if sid in seen:
                raise ValueError(f"Duplicate sample ID: {sid}")
            seen.add(sid)
            filename = Path(row["file_name"])
            if filename.is_absolute() or ".." in filename.parts:
                raise ValueError(f"Unsafe video filename: {filename}")
            prompt = row["prompt"]
            if args.gemini_yxyx:
                if "[xmin, ymin, xmax, ymax]" not in prompt:
                    raise ValueError("Prompt lacks canonical bbox serialization")
                prompt = prompt.replace("[xmin, ymin, xmax, ymax]", "[ymin, xmin, ymax, xmax]")
            record = {"sample_id": sid, "setting": row["metadata"]["setting"],
                      "video_path": filename.as_posix(), "prompt": prompt}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(seen)} model-input records -> {args.out}")


if __name__ == "__main__":
    main()
