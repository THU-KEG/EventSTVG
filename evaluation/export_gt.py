"""Restore the frozen evaluator's GT files directly from integrated release JSONL."""
import argparse
import json
from pathlib import Path

SETTINGS = ('oracle', 'context_1min', 'context_3min', 'context_5min', 'context_10min')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--settings', nargs='+', choices=SETTINGS, default=list(SETTINGS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for setting in args.settings:
        source = args.release_root / 'soccerstvg' / 'videos' / setting / 'metadata.jsonl'
        target = args.out / 'benchmark' / f'{setting}.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        seen = set()
        with source.open(encoding='utf-8') as stream, target.open('x', encoding='utf-8') as output:
            for line in stream:
                row = json.loads(line)
                sid = row['sample_id']
                if sid in seen or sid != row['metadata']['sample_id']:
                    raise ValueError(f'Invalid/duplicate ID: {setting}: {sid}')
                seen.add(sid)
                original = {key: row[key] for key in ['metadata', 'prompt', 'answer']}
                output.write(json.dumps(original, ensure_ascii=False, allow_nan=False) + '\n')
                if setting == 'oracle':
                    spatial_path = args.out / 'spatial' / 'annotations' / f'{sid}_annotation.json'
                    spatial_path.parent.mkdir(parents=True, exist_ok=True)
                    spatial_path.write_text(json.dumps(row['spatial_annotation'], ensure_ascii=False, allow_nan=False) + '\n')
        print(f'{setting}: {len(seen)} records -> {target}')

if __name__ == '__main__':
    main()
