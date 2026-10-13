These utilities prepare or inspect existing labels; they do not introduce automatic annotation.

- `download_data.py`: metadata-only HF access by default, with opt-in exact video IDs and revision pinning. `--dry-run` makes no network request.
- `prepare_inputs.py`: extracts unchanged prompts and relative video associations, excluding answers/provenance from model inputs.
- `verify_examples.py`: checks production prompt/answer formatting and frozen scoring on the five small examples.
- `select_metadata.py`: creates an explicitly identified subset for the historical 300-event comparison.
- `check_release.py`: audits the GitHub payload and optionally exports a video-free ZIP locally.
- `prompt_format.py`: production prompt/answer formatting functions extracted from `bench/generate_benchmark.py`.
- `window_generator/`: existing event/window, correction and quality-control code with portable source roots and tests. It requires original human labels and correction/exclusion manifests; it is not needed to evaluate the release.
- `generate_low_bitrate_contexts.py`: existing resumable FFmpeg context transcoder; no transcode was performed during release preparation.

For future research materialization, pass all source paths explicitly and use a non-date seed:

```bash
PYTHONPATH=scripts python -m window_generator \
  --trimmed-clips source_annotations/trimmed_clips \
  --trim-annotations source_annotations/trim_annotations \
  --role-annotations source_annotations/role_annotation_tracking \
  --soccernet-data source_videos \
  --timestamp-corrections source_annotations/final_timestamp_manifest_3516.jsonl \
  --source-boundary-audit-pass-manifest source_annotations/manual_audit_passes_700.jsonl \
  --source-boundary-tolerance-sec 0.08 \
  --manual-exclusions source_annotations/manual_same_event_duplicate_exclusions.jsonl \
  --output-root data_build --policy-version 8 --seed 42 --manifests-only
```

This would produce new windows; it does not reconstruct the older frozen windows whose historical seed differs. Do not regenerate official GT for evaluation. Frozen manifests and source labels are research/HF materials, not bundled in GitHub. FFmpeg/ffprobe are required for video probing/transcoding. A future encode can use:

```bash
python scripts/generate_low_bitrate_contexts.py --manifest-root data_build \
  --output-root encoded_contexts --contexts context_3min \
  --video-bitrate 2500k --audio-bitrate 128k --preset medium --dry-run
```

Original Stage-1/Stage-2 annotation platforms, cross-annotation platforms, timestamp recovery tools and final temporal review UI were inspected and are indexed in `docs/annotation.md`; they are not duplicated into a new annotation service. Their historic deployment depended on private media roots and institutional task assignments. The runnable public payload focuses on data use and evaluation. No historical reviewer identity/result logs, node modules or deployment credentials are included.
