# Frozen Role-STVG downstream evaluation policy

Policy identifier: `role_stvg_eval_v1.0`

This policy governs validation and scoring after the frozen response parser. It
does not alter `role_stvg_json_extraction_v1`, repair model output, or change
model inference.

## Evaluation pipeline

Formal evaluation uses this pipeline:

```text
raw_response
  -> role_stvg_json_extraction_v1
  -> validation and canonical scoring view
  -> metrics
```

The final record for each `sample_id` is the sole authoritative prediction.
Earlier append-only retry records are retained only for diagnostics.

The evaluator always re-extracts the authoritative record's `raw_response`
with the frozen parser. Persisted `prediction` and `parse_ok` fields are not
trusted as scoring inputs.

The parser extracts exactly one answer-shaped JSON value without repair.
Surrounding prose, a Markdown fence, and extra JSON fields are diagnostics only
and do not reduce a score. Extra fields and extra semantic roles are ignored by
the scoring view. No stage clamps, reorders, fills, interpolates, coerces, or
otherwise repairs model output.

## Component states

Validation preserves component-level states. There is no global `schema_ok`
gate. A temporal window, localization container, row, timestamp, role value,
and bounding box may fail independently.

For a required role at a required timestamp, the canonical prediction state is
one of:

- `box`: an explicitly supplied valid bounding box;
- `null`: an explicitly supplied JSON `null`;
- `missing`: no usable row or no required role key;
- `invalid`: an explicitly supplied malformed role value, or a timestamp made
  unusable by duplication.

`missing` and `invalid` have the same metric value, but remain distinct in
diagnostics. An explicit `null` remains distinct from both.

## Numeric and bounding-box validity

A required numeric value must be a JSON number represented downstream by a
Python `int` or `float`, excluding booleans, and must be finite. `NaN`,
`Infinity`, and `-Infinity` are invalid.

A valid bounding box is a JSON array `[xmin, ymin, xmax, ymax]` satisfying all
of the following:

- length exactly four;
- every coordinate is numeric and finite;
- every coordinate is in the closed interval `[0,1000]`;
- `xmin < xmax` and `ymin < ymax`.

An invalid box invalidates only that role at that timestamp. The evaluator does
not clamp, reorder, or repair coordinates.

## Temporal-window validation for context settings

`temporal_window` must be an object containing finite numeric `start_sec` and
`end_sec`. For input-video duration `D`, it must satisfy

```text
0 <= start_sec <= end_sec <= D.
```

An invalid temporal window produces `tIoU = 0`, joint `vIoU = 0`, and
`vIoU-Int = 0` for the sample. Localization rows never provide a fallback
temporal window.

For a valid predicted interval `P = [p0,p1]` and GT interval `G = [g0,g1]`,
continuous temporal IoU is

```text
intersection = max(0, min(p1,g1) - max(p0,g0))
union        = max(p1,g1) - min(p0,g0)
tIoU         = intersection / union
```

and is zero when `union` is zero.

## Localization-container and row validation

`role_localization` must be a list. A missing, non-list, or empty localization
container does not invalidate a valid temporal window: tIoU is still computed,
while the spatial and joint localization values are zero.

A non-object row or a row with missing, nonnumeric, or nonfinite `time_sec`
invalidates only that row.

Missing required role keys are missing predictions for those roles at those
timestamps. Extra role keys and other extra fields are ignored for scoring and
recorded diagnostically.

An explicit JSON `null` is a valid role value. Whether it is correct is decided
by the metric relative to GT visibility, not by schema validation.

## Timestamp support and canonicalization

Timestamp comparisons use tolerance `1e-6`. Context support alignment retains
the benchmark's nominal 2 Hz index

```text
q(t) = floor(2 * t + 0.5).
```

This is nearest-index rounding with exact half ties rounded upward; it does not
use a language runtime's ties-to-even behavior.

Row order has no scoring effect.

### Oracle

Oracle scoring support is exactly the ordered timestamps listed in the frozen
prompt and GT answer. A missing listed timestamp is a missing prediction.
Timestamps outside that list are ignored for scoring and diagnosed.

### Context

For a valid predicted window, required predicted timestamps are

```text
start_sec, start_sec + 0.5, ... <= end_sec.
```

GT support is the timestamp sequence explicitly present in the frozen GT
`role_localization`. It is not regenerated from the GT interval; this avoids
creating unsupported terminal timestamps.

A prediction row is usable only when its timestamp matches one required
predicted timestamp. Extra, off-grid, and out-of-window timestamps are ignored
for scoring and diagnosed.

If two or more rows match the same required timestamp, that timestamp is
invalid for every role. The evaluator never selects the first or last duplicate.

## Nullable-classic joint role metric

For one required role, let `S_G` and `S_P` be the applicable GT and predicted
temporal supports, and define

```text
D = S_G union S_P
I = S_G intersect S_P
Z = {t in I: GT[t] is explicit null and prediction[t] is explicit null}
```

The nullable extension of classic STVG joint vIoU is

```text
numerator   = sum IoU(GT[t], prediction[t])
              for t in I where both states are valid boxes
denominator = |D| - |Z|
role_score  = numerator / denominator
```

If the adjusted denominator is zero because the complete evaluated role support
is GT-null/predicted-null, the role score is one.

The state contributions are exactly:

- GT box / prediction box: included in the denominator and contributes box IoU;
- GT box / prediction null: included in the denominator with zero numerator;
- GT null / prediction box: included in the denominator with zero numerator;
- GT null / prediction null: excluded from numerator and denominator;
- GT box / prediction missing or invalid: included with zero numerator;
- GT null / prediction missing or invalid: included with zero numerator.

Every timestamp in `S_P - S_G` remains in the denominator, including an
explicitly null prediction. Every timestamp in `S_G - S_P` likewise remains in
the denominator. Thus temporal over-prediction and under-prediction retain the
classic support-union penalty.

GT self-evaluation is one for every required role and sample, including an
entirely invisible role. Predicting null everywhere scores zero for every role
having any visible GT support.

## Visibility-aware intersection role metric

`vIoU-Int` retains the visibility-aware role metric on its intersection domain.
For one required role, define:

- `G`: timestamps where GT has a valid box;
- `P`: timestamps where prediction has a valid box;
- `E`: timestamps where prediction is missing or invalid.

Explicit predicted `null` timestamps are in neither `P` nor `E`. Its score is

```text
numerator   = sum IoU(G[t], P[t]) for t in G intersect P
denominator = |G union P union E|
role_score  = numerator / denominator
```

If GT has no valid box anywhere for the complete role, the intersection role
score is one only when every timestamp in the intersection domain is explicitly
predicted `null`; otherwise it is zero. An empty intersection domain scores
zero.

## Oracle vIoU

For sample `i`, required role `r`, and exact Oracle timestamp domain `T_i`, set
`S_G = S_P = T_i` and use the nullable-classic joint role metric above. Required
roles are averaged equally:

```text
Oracle-vIoU_i = mean(role_score_i,r for r in required_roles_i).
```

## Context joint vIoU and vIoU-Int

Let `S_G` be the nominal-index set of explicit GT localization timestamps and
`S_P` the nominal-index set generated from the valid predicted window.

Joint vIoU uses the nullable-classic role metric with `S_G` and `S_P`, including
the full support-union denominator and the shared explicit-null exclusion. Each
required role is scored separately, then roles are averaged equally.

`vIoU-Int` uses the unchanged visibility-aware intersection role metric on
`S_G intersect S_P`. It is zero when the temporal-support intersection is
empty. Joint vIoU and vIoU-Int are computed independently and are not required
to factorize.

## Diagnostics without score penalties

Diagnostics include, without additional penalties beyond the canonical scoring
states already described:

- prose, Markdown fences, or embedded-answer extraction;
- extra top-level, temporal, row, or semantic-role fields;
- malformed and unavailable components;
- missing, extra, off-grid, duplicate, or out-of-window timestamps;
- missing role keys versus explicit nulls;
- invalid and out-of-range boxes;
- duplicate retry records and non-authoritative attempts.

Diagnostics never repair output and never introduce an additional whole-sample
penalty.

## API response availability

Only an attempt with no received API response is a request failure. HTTP error
responses (including `DataInspectionFailed`) count as returned responses, produce
zero task scores, and remain in the full GT denominator. `N_returned` includes
these responses; `N_error_responses` reports them separately, and `N_request_fail`
counts samples without responses, including missing output records.

Explicit `response_received` takes precedence. Legacy response evidence includes
raw/error response content, the final transport diagnostic HTTP status, or an
SDK `Error code: NNN` message. Earlier retry responses do not establish response
availability for a final transport failure. Model completion parsing and metric
formulas remain unchanged; API errors never supply scored predictions.
