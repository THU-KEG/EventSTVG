"""Frozen public identifiers for the production Role-STVG evaluator."""

EVALUATOR_ID = "role_stvg_eval_v1.0"
EVALUATOR_VERSION = "1.0.0"
PARSER_ID = "role_stvg_json_extraction_v1"

CONTEXT_SETTINGS = (
    "context_1min",
    "context_3min",
    "context_5min",
    "context_10min",
)
SUPPORTED_SETTINGS = ("oracle", *CONTEXT_SETTINGS)
