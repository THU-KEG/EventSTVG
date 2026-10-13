"""Select an explicit ID population for a documented evaluation subset."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--ids", required=True, type=Path, help="JSON array of sample IDs")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    ids = json.loads(args.ids.read_text())
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("IDs must be a JSON array of distinct strings")
    requested = set(ids)
    selected = {}
    for line in args.metadata.open(encoding="utf8"):
        row = json.loads(line)
        if row["sample_id"] in requested:
            if row["sample_id"] in selected:
                raise ValueError("Duplicate GT ID")
            selected[row["sample_id"]] = row
    if requested != selected.keys():
        raise ValueError(f"Missing metadata IDs: {sorted(requested - selected.keys())}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8") as output:
        for sid in ids:
            output.write(json.dumps(selected[sid], ensure_ascii=False) + "\n")
    print(f"Explicit evaluation population: {len(ids)} events -> {args.out}")


if __name__ == "__main__":
    main()
