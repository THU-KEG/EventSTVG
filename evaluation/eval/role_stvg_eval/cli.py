"""Command-line entry point for the complete Role-STVG leaderboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .aggregate import evaluate_model
from .io import discover_predictions, ground_truth_path, load_ground_truth
from .provenance import build_provenance, sha256_file
from .regression import policy_identity_regression
from .render import (
    oracle_role_summary, render_context_split, render_oracle_roles, render_table,
    write_csv, write_json,
)
from .version import EVALUATOR_ID, PARSER_ID, SUPPORTED_SETTINGS


SETTING_ALIASES = {
    "1min": "context_1min",
    "3min": "context_3min",
    "5min": "context_5min",
    "10min": "context_10min",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument(
        "--setting",
        choices=("all", *SUPPORTED_SETTINGS, *SETTING_ALIASES),
        default="all",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def run_setting(setting: str, args: argparse.Namespace) -> dict:
    internal = SETTING_ALIASES.get(setting, setting)
    artifact_dir = args.out.resolve() / internal
    artifact_dir.mkdir(parents=True, exist_ok=True)
    gt_root = args.gt.resolve()
    gt = load_ground_truth(gt_root, internal)
    sources = discover_predictions(args.outputs.resolve(), internal)
    all_scores, all_samples, all_roles, diagnostics = [], [], [], []
    for source in sources:
        scores, samples, roles, diagnostic = evaluate_model(gt, source)
        all_scores.append(scores)
        all_samples.extend(samples)
        all_roles.extend(roles)
        diagnostics.append(diagnostic)
    write_csv(artifact_dir / "scores_raw.csv", all_scores)
    write_json(artifact_dir / "scores.json", {
        "evaluator_id": EVALUATOR_ID,
        "parser_id": PARSER_ID,
        "setting": internal,
        "scores": all_scores,
    })
    write_csv(artifact_dir / "per_sample.csv", all_samples)
    write_csv(artifact_dir / "per_role.csv", all_roles)
    write_json(artifact_dir / "diagnostics.json", {
        "evaluator_id": EVALUATOR_ID,
        "parser_id": PARSER_ID,
        "setting": internal,
        "models": diagnostics,
    })
    table = render_table(all_scores, internal, artifact_dir)
    role_table = None
    if internal == "oracle":
        role_scores = oracle_role_summary(all_roles)
        write_csv(artifact_dir / "role_scores_raw.csv", role_scores)
        write_json(artifact_dir / "role_scores.json", {"setting": internal, "scores": role_scores})
        role_table = render_oracle_roles(role_scores, artifact_dir)
    split_tables = (
        render_context_split(all_scores, artifact_dir, internal)
        if internal != "oracle" else None
    )
    regression = policy_identity_regression(gt, internal)
    write_json(artifact_dir / "regression.json", regression)
    if not regression["passed"]:
        raise RuntimeError(f"{internal} regression failed; see {artifact_dir / 'regression.json'}")
    provenance_path = artifact_dir / "provenance.json"
    write_json(provenance_path, build_provenance(gt_root, internal, gt, sources))
    return {
        "evaluator_id": EVALUATOR_ID,
        "parser_id": PARSER_ID,
        "setting": internal,
        "gt_records": len(gt),
        "models": [source.model for source in sources],
        "prediction_files": [str(source.path.resolve()) for source in sources],
        "artifact_dir": str(artifact_dir),
        "provenance_file": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "table": table,
        "role_table": role_table,
        "split_tables": split_tables,
        "regression_passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.setting == "all":
        settings = tuple(
            setting for setting in SUPPORTED_SETTINGS
            if ground_truth_path(args.gt.resolve(), setting).is_file()
        )
    else:
        settings = (args.setting,)
    manifest = [run_setting(setting, args) for setting in settings]
    write_json(args.out.resolve() / "manifest.json", {
        "evaluator_id": EVALUATOR_ID,
        "parser_id": PARSER_ID,
        "runs": manifest,
    })
    print(args.out.resolve() / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
