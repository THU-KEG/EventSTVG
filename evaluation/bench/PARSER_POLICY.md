# Frozen Role-STVG JSON extraction policy

Policy identifier: `role_stvg_json_extraction_v1`

The parser decides only whether a JSON prediction can be extracted from an API
response. API success, protocol-format compliance, and benchmark evaluation are
separate stages.

For the relevant setting, the parser:

1. accepts a complete stripped response that is valid, answer-shaped JSON;
2. accepts a whole-response Markdown `json` or unlabeled code fence containing
   one valid, answer-shaped JSON value;
3. otherwise structurally locates complete JSON values in surrounding prose and
   accepts only one answer-shaped occurrence;
4. rejects zero occurrences and rejects two or more occurrences, including
   repeated equal answers and self-corrections.

Basic answer shape:

- `oracle`: a non-empty top-level array whose elements are localization-row
  objects containing `time_sec`;
- `oracle_with_bbox`: the same top-level array shape as `oracle`; role values
  are frame-local candidate IDs rather than coordinate arrays, but value-level
  compliance remains outside the extraction policy;
- `context_1min`: a top-level object containing `temporal_window` and
  `role_localization`.

The parser never repairs JSON, changes model content, selects the first/last/best
answer, or uses timestamp bounds, cadence, role correctness, bbox validity,
temporal accuracy, ground truth, or benchmark scores. Those checks belong to a
separate protocol-format diagnostic or evaluator.
