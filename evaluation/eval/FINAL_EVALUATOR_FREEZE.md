# Final Role-STVG evaluator freeze

## Freeze verdict

`role_stvg_eval_v1.0` is the frozen Role-STVG downstream evaluator. Its package
version is `1.0.0`, and its frozen parser is
`role_stvg_json_extraction_v1`. The normative metric and validation semantics
are defined by `EVAL_POLICY.md`.

The evaluator supports `oracle`, `context_1min`, `context_3min`,
`context_5min`, and `context_10min` through one CLI. The context aliases
`1min`, `3min`, `5min`, and `10min` resolve to the corresponding canonical
setting names.

The five-model `context_1min` reproduction from existing raw responses matches
the finalized nullable-classic-v1 reference exactly in every score field. No
inference was rerun.

## Frozen identifiers and source hashes

| Item | Value | SHA-256 |
|---|---|---|
| Evaluator ID (`VERSION`) | `role_stvg_eval_v1.0` | `4e3655674268d4287f9f63148777eeb0ca57f35f201d3e03c9392203efaae82f` |
| Package version (`pyproject.toml`) | `1.0.0` | `923a02946e39f99b345c01aefb93cac2f232b17b6981f30cc8f1449ed32bd7ba` |
| Parser ID | `role_stvg_json_extraction_v1` | parser source: `ec42515741c80e7d66704ce376fb87abc75faf114a766d7792d7263a7ca0ccc8` |
| Evaluation policy (`EVAL_POLICY.md`) | `role_stvg_eval_v1.0` | `8610f003aa7eaed0dcab1e736619e63f610e6bb593a1a0ceb26deedb7b4bab63` |

The hashes of every evaluator source module are recorded in the reproduction
provenance file:

`research/eval/artifacts_freeze_reproduction_v1_0_diagnostics_fix/context_1min/provenance.json`

## Reference reproduction

The reproduction command, run from `research/eval`, was:

```bash
python -m role_stvg_eval.cli \
  --outputs ../bench/outputs \
  --gt ../bench/soccerrole \
  --setting context_1min \
  --out artifacts_freeze_reproduction_v1_0_diagnostics_fix
```

Reference directory:

`research/eval/artifacts_nullable_classic_v1/formal/1min`

Reproduction directory:

`research/eval/artifacts_freeze_reproduction_v1_0_diagnostics_fix/context_1min`

### Exact score check

| Model | N | m-tIoU | m-vIoU | m-vIoU-Int |
|---|---:|---:|---:|---:|
| GLM-4.6V | 3516 | 0.06346385954393362 | 0.001308345610105149 | 0.003246152670738934 |
| GLM-5.3-Flash | 3516 | 0.3137920736095246 | 0.007427877180177431 | 0.015018577865545679 |
| MiniMax-M3 | 3516 | 0.04179286153400564 | 0.0002115235951442636 | 0.0005144099055250597 |
| Qwen3.7-Plus | 3516 | 0.38471732299370726 | 0.004218844918899402 | 0.008122217738848 |
| Qwen3.8-Max | 3516 | 0.4645769234252286 | 0.03952635394425306 | 0.07242959054350817 |

All 23 aggregate fields for all five models are exactly equal to the reference,
including all strict-threshold rates. All sample metric values in all 17,580
per-sample rows and all eight shared fields in all 32,485 per-role rows are
also exactly equal and in the same order. Validation diagnostics intentionally
add the scoring-neutral `extra_top_level_fields` count where applicable.
Regression results are equal except that the frozen output correctly records
the versioned policy ID
`role_stvg_eval_v1.0` instead of the older label `role_stvg_eval_v1`; both pass.

The files are not expected to be byte-identical because the frozen artifacts
add evaluator/parser IDs and provenance fields without changing scores.

### Artifact hashes

Finalized nullable-classic-v1 reference:

| Artifact | SHA-256 |
|---|---|
| `scores.json` | `169a36f88484e1caf4fb8cd1856da982048cd932744f9be55548c386b9cd268b` |
| `scores_raw.csv` | `3a1552eca3aef80f029ae5209a77a482b670ec8809036580f3ce658d2f4fd1e2` |
| `per_sample.csv` | `86a76019d9479cb9d4404090d5e8558f663a2f6a451f233d442ce6cf0cb1764d` |
| `per_role.csv` | `dfce56647013c4d6aa99ae3a76374e1b40b502e7c79c0b35ed7957c2bd179245` |
| `diagnostics.json` | `8cc6ad48c64a6299f57a8b5d55c17e671caa9201a812e4346a5adc60fed7cfdf` |
| `regression.json` | `2f147e834fbc844ddca175aa5bbd95b6267c90b76491d3f58d9a3bc4a6ab38ae` |

Frozen evaluator reproduction:

| Artifact | SHA-256 |
|---|---|
| root `manifest.json` | `015d7a800892cea753057f539b05eac25c17163867d1cfa3225f5d5cea1f61ef` |
| `context_1min/scores.json` | `0cda997932a4f709b241b03a1fb2e9f4125cb7aefed31c92c6c2895f01e4051a` |
| `context_1min/scores_raw.csv` | `074dfb4a80d75880d09f717b612b4c26cddd925bcb41ebb11af3c3c2a12a31a1` |
| `context_1min/per_sample.csv` | `dafb97d504e4d5728b5e241ac8ed93eeca95929757e91fcbcec69085369ae6c9` |
| `context_1min/per_role.csv` | `9d0e2a5c008a6d5c2f9fe0fdf03421135e36e7ff8a2c9a540db2f82a6926fc50` |
| `context_1min/diagnostics.json` | `839f8e2ba8a2e1d31d6855325f707601c362973e5f273f1f1b3866099a56cab5` |
| `context_1min/regression.json` | `7d425863a502270caff140493adfda6b5c1fdcc9db46a0ed6a62c2aa6c3d2a9a` |
| `context_1min/provenance.json` | `50cc3814251ebbd9318366b3fdbe9ee725e7bf197c46cdeb904d1adf569e4bee` |

### Frozen input hashes

| Input | SHA-256 |
|---|---|
| `bench/soccerrole/benchmark/context_1min.jsonl` | `562919e16b4ec93d3a05e1c4a7980b6c4986d2f08cb158b31e3d7daf5587dc97` |
| `bench/outputs/GLM-4.6V/context_1min.jsonl` | `4d5bbc8753396c4b377fe18ba3d0ddb8bcb44b0c9a469a75f8792b82b2aff0a0` |
| `bench/outputs/GLM-5.3-Flash/glm53flash_context_1min.jsonl` | `b99827a65c4f216957ce2e9e883f5d1118e7096f0910e85fd0a86526df75a870` |
| `bench/outputs/MiniMax-M3/context_1min.jsonl` | `f5cfb2b676a90bf7d35161242c14c6098ec52930bce3716949c0b89ab2549ebd` |
| `bench/outputs/Qwen3.7-Plus/context_1min.jsonl` | `89723f83a7cddaf5f75b5dd089d469e2732227b4517d8abeef968d6c5712c65f` |
| `bench/outputs/Qwen3.8-Max/context_1min.jsonl` | `e97ea0fb1b1f81669c481881edeec0cd50681083dc395af2ae554f98e8a924fb` |

## Final validation

The complete test suite passes: `52 passed`.

## Canonical future commands

Run these commands from `research/eval`. Replace
`<ARTIFACT_ROOT>` with a new output directory; do not reuse an existing formal
artifact directory.

```bash
python -m role_stvg_eval.cli --outputs ../bench/outputs --gt ../bench/soccerrole --setting oracle --out <ARTIFACT_ROOT>
python -m role_stvg_eval.cli --outputs ../bench/outputs --gt ../bench/soccerrole --setting context_1min --out <ARTIFACT_ROOT>
python -m role_stvg_eval.cli --outputs ../bench/outputs --gt ../bench/soccerrole --setting context_3min --out <ARTIFACT_ROOT>
python -m role_stvg_eval.cli --outputs ../bench/outputs --gt ../bench/soccerrole --setting context_5min --out <ARTIFACT_ROOT>
python -m role_stvg_eval.cli --outputs ../bench/outputs --gt ../bench/soccerrole --setting context_10min --out <ARTIFACT_ROOT>
python -m role_stvg_eval.cli --outputs ../bench/outputs --gt ../bench/soccerrole --setting all --out <ARTIFACT_ROOT>
```

For reproducibility, preserve the generated root `manifest.json` and each
setting's `provenance.json` together with its scores and row-level artifacts.
