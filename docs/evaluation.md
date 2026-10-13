# Official football evaluation

The released implementation is the existing `role_stvg_eval_v1.0` (package 1.0.0) and parser `role_stvg_json_extraction_v1`. Freeze records, experiment calls and reproduction scores identify this implementation rather than the older `bench/evaluate_oracle_viou.py`, parse-success-only analyses or diagnostic scripts. All metric/parser source modules are copied byte-for-byte. Relative parser location `evaluation/bench/parser_policy.py` is deliberately retained. `evaluation/evaluate.py` is a new standalone file-interface wrapper over the same loader and `evaluate_model`, not a second metric implementation.

From the release root:

```bash
python evaluation/evaluate.py --setting oracle \
  --metadata examples/oracle.metadata.jsonl \
  --predictions examples/predictions/copy_gt/oracle.jsonl \
  --out artifacts/oracle.json
python evaluation/evaluate.py --setting context_3min \
  --metadata examples/context_3min.metadata.jsonl \
  --predictions examples/predictions/copy_gt/context_3min.jsonl \
  --out artifacts/context_3min.json
```

Use `context_1min`, `context_5min` and `context_10min` similarly, matching the metadata setting. Each setting must be scored against its own prompt/time origin. Complete metadata is the population: the wrapper does not filter GT to predicted IDs. Explicitly evaluate a documented 300-ID subset only when making the matched long-context comparison; it is not a 3,516-event leaderboard score.

Prediction JSONL rows use the exact sample ID:

```json
{"sample_id":"UNCHANGED_EVENT_ID","status":"ok","raw_response":"MODEL_JSON_ANSWER_AS_A_STRING"}
```

`raw_response` is re-extracted by the frozen parser. Persisted `prediction`/`parse_ok` fields never override it. Last JSONL row for a repeated ID wins, including a final failed attempt; retries are retained for diagnostics. Extra IDs cannot improve the GT score. A completely empty prediction file is supported by the compact wrapper and produces zeros on all GT samples. A malformed JSONL envelope, missing string `sample_id`, duplicate GT ID, wrong setting or invalid GT structure raises an error; these are input file errors rather than model-answer failures scored zero.

Oracle raw answers are localization arrays; context raw answers are objects containing `temporal_window` and `role_localization`. Times are input-video seconds. Boxes are normalized xyxy in `[0,1000]`, finite and positive extent. No clamping, coordinate reordering, filling, interpolation or numeric coercion is applied. Semantic roles are matched only with the same role key; no cross-role permutation is allowed.

**tIoU** is continuous interval intersection duration divided by interval union duration. Predicted windows must satisfy `0 <= start_sec <= end_sec <= input_duration`. Invalid/missing temporal windows score zero for all context metrics. A valid temporal window with unavailable spatial rows retains its temporal score. Component failures invalidate only affected role/frame entries where possible.

Oracle support is the explicitly stored GT/prompt timestamp list. Context GT support is its explicit localization list; prediction support is generated from predicted start at `0.5`-second intervals through predicted end. Row matching uses tolerance `1e-6`, and nominal context indices are `floor(2*t+0.5)`. Off-grid/out-of-window/extra rows are ignored and diagnosed. Duplicate required timestamps invalidate that timestamp for all roles. Row order does not change scores.

For a role, let `S_G/S_P` be GT/prediction support, and `Z` their shared timestamps where both values are explicit null. **Joint vIoU** sums ordinary box IoU where both boxes are valid, dividing by `|S_G union S_P| - |Z|`. A zero adjusted denominator scores one. Missing/invalid predictions are penalized even when GT is null; matched explicit null pairs are excluded. Over/underextended support remains in the denominator, including predicted null outside GT support.

**vIoU-Int** uses the visibility-aware role metric on the support intersection. Its denominator combines GT-visible timestamps, prediction-visible timestamps and missing/invalid predictions. An entirely invisible GT role scores one only when every intersecting prediction is explicitly null; an empty intersection scores zero. It is computed separately, not as `vIoU/tIoU`. Oracle vIoU uses the nullable-classic metric on the exact Oracle domain.

Aggregate support contributions within each role, average required roles within each sample, then average samples equally. Report `m_tIoU`, `m_vIoU`, `m_vIoU_Int` and proportions with strict `score > threshold` for `0,0.1,0.3,0.5`. Stored results are fractions; historical table CSVs display percentages. Do not conflate `>` with `>=` or replace failure-retaining means with parse-success-only means.

Missing predictions, service errors and unparseable final outputs remain in `N` and score zero. Extra prose or Markdown fences can be extracted without repair and are diagnostic-only. API error responses count as returned responses where evidence establishes receipt, but never provide a scored answer; missing/no-response attempts remain request failures. These diagnostic counts were updated in the existing source after the original freeze report; task metric behavior is unchanged. Full normative definitions are in [EVAL_POLICY.md](../evaluation/eval/EVAL_POLICY.md).

For original leaderboard reports, export frozen metadata to the original GT layout, then run:

```bash
python evaluation/export_gt.py --release-root data --settings oracle context_1min \
  --out evaluation_gt
PYTHONPATH=evaluation/eval python -m role_stvg_eval.cli \
  --outputs examples/predictions --gt evaluation_gt --setting context_1min \
  --out artifacts/leaderboard
```

Replace the examples prediction directory with real predictions for a full run. Predictions are discovered as `<outputs>/<model>/<setting>.jsonl` or a single `*_<setting>.jsonl` per model; ambiguous files cause an error. `--setting all` evaluates available GT settings separately and does not pool events. Oracle additionally requires original spatial annotations; both export utility and compact wrapper restore those from `spatial_annotation`. The exporter never repairs labels.

Run frozen tests with `cd evaluation/eval && pytest -q`, and runnable example validation with `python scripts/verify_examples.py`. The sample tests include ordinary/multi-role data, explicit timestamp support, missing predictions, invalid raw responses and duplicate retry histories.
