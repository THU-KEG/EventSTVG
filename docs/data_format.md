# Football data format and access

The [HF dataset](https://huggingface.co/datasets/zacharyz21/EventSTVG) uses one integrated metadata file for each setting:

```text
soccerstvg/videos/
  oracle/metadata.jsonl
  context_1min/metadata.jsonl
  context_3min/metadata.jsonl
  context_5min/metadata.jsonl
  context_10min/metadata.jsonl
```

Videos are stored alongside each setting's metadata at the relative `file_name` path, organized by league/season/match/sample. Each JSONL has 3,516 unique IDs. All five ID sets are identical. A role instance is an event–role pair, not a unique person; multiple setting instances share the same target and original spatial annotation. The settings are evaluation configurations, all historically named `test`, with no verified training/validation split. Context windows need not be nested.

| Field | Contract |
| --- | --- |
| `sample_id` | Stable target event ID; join predictions by exact string, never row index |
| `file_name` | MP4 path relative to this `metadata.jsonl` directory |
| `prompt` | Complete stored model-input prompt, including requested roles and output schema |
| `event_type`, `active_roles` | Final semantic event type and authoritative requested-role list |
| `metadata` | Original sample, source, timing, event order, setting and version provenance |
| `answer` | Oracle localization array; context object with `temporal_window` and `role_localization` |
| `spatial_annotation` | Original pixel annotations, dimensions, FPS, frames and per-role tracking records |

Every localization row contains `time_sec` plus each requested role key. Role values are `[xmin,ymin,xmax,ymax]` normalized to `[0,1000]`, or explicit JSON `null`. X scales with original width and Y with original height. Boxes use top-left image origin. Original annotations use pixel xyxy coordinates and zero-based event-clip frame indices.

Answer times are seconds from the **input video start**, not the match clock. Oracle starts at the trimmed event clip origin. Context `answer.temporal_window` and `role_localization[].time_sec` share the context-video origin. `metadata.window_start/window_end` are source-half-video times; `target_start_in_window/target_end_in_window` are context-local times. Their sum with window origin traces the same target across settings.

The nominal localization support is 2 Hz, with offsets `k*0.5` from target start. Preserve the **explicit stored timestamp list**; do not interpolate or invent a terminal frame. Original keyframes use half-up rounding of `fps_original*0.5*k`; at 25 FPS this gives `0,13,25,38,...`. Original frame time may be `0.52` while answer time is nominal `0.5`. Explicit `null` means no reliable localization in the exported label; it differs from absent/malformed model output. The historical exporter also inserts null when a role has no annotation entry at a keyframe, so data fields alone do not prove every null was separately adjudicated as invisible.

Only video and prompt enter the model request. Do not supply `answer`, `spatial_annotation`, target timing or ordinal provenance as additional model context. The requested ordinal already appears in the official prompt. Use:

```bash
python scripts/prepare_inputs.py \
  --metadata data/soccerstvg/videos/oracle/metadata.jsonl \
  --out runtime/oracle.jsonl
```

This extracts exactly `sample_id`, `setting`, relative `video_path` and unchanged `prompt`. It does not generate annotations. Prepare each setting into a separate JSONL. Gemini's historical yxyx prompt adaptation is explicit via `--gemini-yxyx`; it is a separate baseline adaptation, not the HF official answer format.

The dataset is [zacharyz21/EventSTVG](https://huggingface.co/datasets/zacharyz21/EventSTVG). Request access on that page and authenticate using an approved account. Pin an HF commit revision for reproducible evaluation:

```bash
python -m pip install -r requirements.txt
hf auth login
python scripts/download_data.py --repo-id zacharyz21/EventSTVG \
  --revision HF_COMMIT_SHA --setting oracle --local-dir data
python scripts/download_data.py --repo-id zacharyz21/EventSTVG \
  --revision HF_COMMIT_SHA --setting oracle --local-dir data \
  --sample-id 'EXACT_EVENT_ID'
```

The first command downloads one metadata file. The second also downloads the selected MP4. Authentication uses the installed HF client's normal credentials; credentials are never bundled. For local metadata inspection, Python's standard `json` module needs no video backend. VideoFolder loading after selected downloads is optional:

```python
from datasets import load_dataset, Video
ds = load_dataset("videofolder", data_files={"test": [
    "data/soccerstvg/videos/oracle/**/*.mp4",
    "data/soccerstvg/videos/oracle/metadata.jsonl",
]}, split="test", drop_labels=True)
ds = ds.cast_column("video", Video(decode=False))
```

A partial video download does not constitute the full evaluation population. For scoring use the complete setting metadata, even when only some predictions exist. If using HF's heterogeneous role structs, rely on `active_roles`; schema-added null fields are not additional labels.

Use `file_name` or the loaded `video` feature for media access. Stored source provenance is descriptive; it is not a runtime filesystem dependency. The GitHub examples use relative paths and preserve prompts, answers and spatial labels.
