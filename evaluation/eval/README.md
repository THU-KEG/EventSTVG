# Role-STVG Evaluation Suite

This package evaluates the frozen Role-STVG Oracle and one-minute benchmark
outputs under [EVAL_POLICY.md](EVAL_POLICY.md). It reads prediction JSONLs
without modifying them and writes evaluation products under the requested
output directory.

## Inputs

- Ground truth: `soccerrole/benchmark/oracle.jsonl` and
  `soccerrole/benchmark/context_1min.jsonl`.
- Predictions: `bench/outputs/<model>/oracle.jsonl` and
  `bench/outputs/<model>/context_1min.jsonl`.

Formal output files are append-only retry histories. The last record for each
`sample_id` is authoritative. Its `raw_response` is re-extracted with the
frozen parser; persisted parse fields are not scoring inputs. Duplicate IDs and
extra retry records remain visible in diagnostics. A missing, service-failed,
or unparseable final prediction stays in the 3,516-sample denominator.

## Role identity

> Semantic roles are evaluated by identity. Cross-role permutation matching is never allowed.

For required roles \(\mathcal R_i\), role \(r\) is compared only with predicted
role \(r\). Missing roles score zero; extra roles are diagnosed and cannot
improve scores. Scores aggregate frame → role → sample → dataset, so every
sample has equal dataset weight.

## Metrics

Box IoU is ordinary xyxy intersection over union after strict finite,
positive-extent, `[0,1000]` validation. Oracle support is exactly the
timestamps in the prompt/GT answer. Context GT support likewise comes from its
explicit localization rows; predicted support is generated at 0.5-second
cadence from a valid predicted window.

For one-minute temporal windows \(T^p,T^g\):

\[
tIoU = |T^p \cap T^g| / |T^p \cup T^g|.
\]

The official tIoU is continuous. For joint context vIoU, let `S_G` and `S_P`
be the explicit GT and generated prediction supports, and let `Z_r` be their
intersection timestamps where role `r` is explicit null in both GT and the
prediction. The nullable-classic role value is:

\[
vIoU_r =
\frac{\sum_{t\in S_G\cap S_P,\;B^g_r(t),B^p_r(t)\ valid} IoU(B^p_r(t),B^g_r(t))}
{|S_G\cup S_P|-|Z_r|}.
\]

GT-null/predicted-null contributes to neither numerator nor denominator within
the shared support. Timestamps outside either support remain in the temporal
union denominator, including predicted-null over-extension. A zero adjusted
denominator scores one. vIoU-Int retains the visibility-aware role metric on
the support intersection and is zero when that intersection is empty. No
interpolation or output repair is performed.

Interpretation:

- **tIoU** = temporal event localization.
- **vIoU** = joint temporal + spatial Role-STVG performance.
- **vIoU-Int** = spatial role-grounding quality inside the temporally overlapping region.

All reported threshold proportions use strict `score > threshold` for
`0, 0.1, 0.3, 0.5`. Values remain in `[0,1]` in JSON/CSV and are converted to
percentages only when tables are rendered.

## Run

From the `eval/` directory:

```bash
python -m role_stvg_eval.cli \
  --outputs ../bench/outputs \
  --gt ../bench/soccerrole \
  --setting all \
  --out artifacts
```

Run tests with:

```bash
pytest -q
```

Each setting produces raw scores, per-sample and per-role CSVs, diagnostics,
regression results, LaTeX, PDF, and 300-DPI PNG tables. If LaTeX is unavailable,
the `.tex` source is preserved and PDF/PNG previews are rendered deterministically
with Matplotlib.

Oracle additionally produces `role_scores_raw.csv`, `role_scores.json`, and
`oracle_role_scores.{tex,pdf,png}` with m-vIoU broken down by semantic role.

The one-minute leaderboard is also rendered as separate temporal-grounding
`one_min_tg_scores.{tex,pdf,png}` and spatial-grounding
`one_min_sg_scores.{tex,pdf,png}` tables. The spatial view is additionally
split into standard joint `one_min_stvg_scores.{tex,pdf,png}` and
intersection-only `one_min_viou_int_scores.{tex,pdf,png}` tables.
