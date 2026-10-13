# Football baseline reproduction materials

The released baseline code derives from the existing research runners, with path relocation documented in `docs/evidence/source_manifest.json`. It sends stored prompts, never annotations. Closed-source APIs have not been called during this preparation. Model names below are **recorded experimental aliases**, not assertions about current public availability or checkpoint equivalence. Historical experiment dates, when recorded, are retained as UTC timestamps in [recorded_runs.json](configs/recorded_runs.json); absent dates are unknown, not inferred from file names or modification times.

| Baseline route | Recorded model IDs | Visual input / generation |
| --- | --- | --- |
| OpenAI-compatible gateway | `Qwen3.8-Max`, `Qwen3.7-Plus`, `MiniMax-M3`, `GLM-4.6V`, `glm-5.3-flash` | Original MP4, base64 `video_url`; client does not choose FPS or resize; optional generation fields omitted |
| OpenRouter frames | `openai/gpt-5.6-sol`, pinned `OpenAI` provider | 2 FPS; cap 1,200 frames; 512×288; JPEG quality 80; detail `low`; generation defaults |
| OpenRouter native video | `qwen/qwen3.5-9b`, provider `SiliconFlow` in recorded route | Original MP4; no client FPS/resize; no generation overrides |
| Local Qwen | `qwen3.5-9b`, local `Qwen3.5-9B` | Native processor sampling/resize defaults; `enable_thinking=False`, greedy generation, `max_new_tokens=4096` |
| PTD | released PTD Qwen3-VL-4B; `Role-STVG-PTD-64f-v2` | One role per sequential request; 64 uniform distinct frames; FP16/SDPA; pixels 131,072–786,432; temperature 0; max 1,024 new tokens |
| Recorded Gemini | `gemini-3.7-flash` | Google JPEG-frame route and adapted yxyx prompt; see limitations below |

The full prompt specification is [task_spec.json](../examples/task_spec.json), with full sample prompts in `examples/*.metadata.jsonl`. Role definitions and Oracle timestamp lists must remain sample-specific. The historical numbered-box Oracle and PTD referring-query controls are distinct input protocols and must not be confused with direct box prediction. PTD's 33 event–role query templates are retained in `ptd/policy/role_query_templates.json`.

API dependency installation and runtime inputs, from the release root:

```bash
python -m pip install -r baselines/requirements-api.txt
python scripts/prepare_inputs.py \
  --metadata data/soccerstvg/videos/context_1min/metadata.jsonl \
  --out runtime/context_1min.jsonl
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808
export ALL_PROXY=http://127.0.0.1:10808
export http_proxy=http://127.0.0.1:10808
export https_proxy=http://127.0.0.1:10808
export all_proxy=http://127.0.0.1:10808
export NO_PROXY=
export no_proxy=
```

Supply your own authorized credentials in a protected external env file (`chmod 600`): `SCHOOL_API_KEY/SCHOOL_BASE_URL` for the gateway, or `OPENROUTER_API_KEY/OPENROUTER_BASE_URL` for OpenRouter. No credential files, institutional accounts or signed URLs are included. Commands below incur inference only when deliberately executed with valid credentials; none were executed here beyond CLI help and mocked tests.

```bash
PYTHONPATH=baselines python -m inference_local.run \
  --setting context_1min --model Qwen3.8-Max --manifest-dir runtime \
  --video-root data/soccerstvg/videos/context_1min \
  --env-file credentials.env --workers 1 \
  --output outputs/Qwen3.8-Max/context_1min.jsonl

PYTHONPATH=baselines python -m inference_openrouter.run \
  --setting context_1min --model openai/gpt-5.6-sol --input-mode frames \
  --manifest-dir runtime --video-root data/soccerstvg/videos/context_1min \
  --env-file credentials.env --fps 2 --max-frames 1200 --image-detail low \
  --output outputs/GPT-5.6-Sol/context_1min.jsonl
```

Prepare Oracle in `runtime/oracle.jsonl` and use its video directory for native Qwen:

```bash
PYTHONPATH=baselines python -m inference_openrouter.run \
  --setting oracle --model qwen/qwen3.5-9b --input-mode native-video \
  --provider-order SiliconFlow --manifest-dir runtime \
  --video-root data/soccerstvg/videos/oracle --env-file credentials.env \
  --output outputs/Qwen3.5-9B-OpenRouter/oracle.jsonl
```

Gateway/OpenRouter orchestration preserves append-only retries and raw model responses. Only explicit reruns should replace a completed response with a later authoritative attempt. The local-Qwen runner has a different historical retry mechanism: `--retry-failures` rewrites failed records; do not claim it preserves append-only histories.

Local Qwen additionally needs PyTorch, a Transformers build exposing `AutoModelForMultimodalLM`, the native video processor dependencies and authorized local weights. The successful CLI-help environment observed here uses PyTorch `2.4.1+cu121` and Transformers `5.13.0.dev0`; the development-build commit and model checkpoint revision are not established. The older `stvg` environment lacks that auto-model class. Imports were deferred until model load so help works without the class. No new architecture mapping or fallback was substituted.

```bash
PYTHONPATH=baselines python -m inference_local_models.runner \
  --model qwen3.5-9b --model-path models/Qwen3.5-9B \
  --manifest runtime/oracle.jsonl --video-root data/soccerstvg/videos/oracle \
  --output outputs/Qwen3.5-9B/oracle.jsonl
```

PTD code/weights are external. Set `PTD_CODE_ROOT` to the upstream source root and `PTD_MODEL_PATH` to the checkpoint directory, using that model's compatible environment (PyTorch, Decord, Transformers and qwen-vl-utils). Prepare Oracle and context_1min model-input manifests. Then:

```bash
export PTD_CODE_ROOT=models/ParallelTubeDecoding
export PTD_MODEL_PATH=models/ParallelTubeDecoding-Qwen3-VL-4B
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python baselines/ptd/run_formal.py \
  --setting oracle --manifest-dir runtime \
  --video-base-dir data/soccerstvg/videos --output-root outputs/PTD
```

Run context_1min sequentially with the same flags. The port preserves generation and sampling but relocates paths, manifests and output destinations; its `SHA256SUMS` validates the release integration files, not the historical full-prompt files. Original hashes are retained in provenance. It is not a new freeze of identical historical filesystem inputs. PTD protocol frozen time is `2026-09-07T12:41:15.538956+00:00`. Model inference has not been rerun or GPU-validated in this preparation.

PTD outputs native time/box tokens. Use the existing offline adapter (no model/API):

```bash
python evaluation/export_gt.py --release-root data \
  --settings oracle context_1min --out evaluation_gt
python baselines/ptd/eval/evaluate_ptd.py --ptd-root outputs/PTD \
  --gt-root evaluation_gt --evaluator-parent evaluation/eval --setting oracle
```

This delegates task scoring to the frozen evaluator. Its native-frame Voronoi adaptation, sampling provenance requirements and per-role merge are preserved rather than treating PTD tokens as direct xyxy JSON.

**Gemini limitation:** the exact Google inference runner and complete generation configuration were not located in the inspected sources. Existing one-minute frame records show 2 FPS, 512×288, JPEG quality 85, cap 900, and optional `max_output_tokens/response_mime_type`; later Oracle/3/5/10-minute records do not retain the complete media settings, and report no optional generation fields. Thus these details cannot be generalized to every Gemini run. One-minute base and retry32k histories were merged using last-record precedence. Original responses use yxyx; the research scoring path converts only baseline coordinates to xyxy before frozen scoring. This is a declared serialization adapter, not an evaluator repair. `adapt_gemini.py` retains original raw responses and writes a separate adapted prediction file. The public integration does not invent a new Google request protocol or claim exact end-to-end reproduction.

Historical tables are copied as compact evidence in [results/](results/). They report percentages and have two different populations: Oracle/1-minute generally 3,516 events, while the main 3/5/10-minute comparison uses a matched **300-target panel selected with seed 21**. Its IDs are in [long_context_300_ids.json](configs/long_context_300_ids.json). Select metadata explicitly before scoring the panel; absent predictions count zero within that selected GT. Seed 42 is used for any new reproducible random work.

**GLM-5V-Turbo Oracle is an exception:** the table's mean vIoU `10.32%` is a frozen projection to a 3,516 target population; threshold rates are measured on 934 executed queries with failures retained. `oracle_observed_N`, `oracle_vIoU_N`, `oracle_recall_N` and score-type columns expose the distinction. It is not a complete measured Oracle run. The observed summary and estimate provenance are retained separately. Publication wording and whether to present that estimate require author confirmation.

All five legacy gateway one-minute baselines were rescored from existing raw responses during this preparation and reproduced historical task score fields exactly. No paid inference was executed. Client preprocessing, response adaptation and failures differ across routes, so these experiments are not a controlled model-architecture comparison. Exact backend revisions, absent experimental timestamps, Google runner/configuration, and some local model environment pins remain reproduction gaps.
