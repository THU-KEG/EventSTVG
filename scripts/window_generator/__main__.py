from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generator import CONTEXTS, generate_all, SUPPORTED_POLICY_VERSIONS
from .corrections import apply_corrections
from .quality_control import load_manual_exclusions
from .sources import (
    Roots,
    build_event_index,
    discover_trim_inventory,
    source_coverage,
    validate_roots,
)


ROLESTVG_ROOT = Path("data_build")
SOCCERNET_PROJECT = Path("source_annotations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic RoleSTVG oracle/context datasets."
    )
    parser.add_argument("--trimmed-clips", type=Path,
                        default=SOCCERNET_PROJECT / "trimmed_clips")
    parser.add_argument("--role-annotations", type=Path,
                        default=SOCCERNET_PROJECT / "role_annotation_tracking")
    parser.add_argument("--trim-annotations", type=Path,
                        default=SOCCERNET_PROJECT / "trim_annotations")
    parser.add_argument("--soccernet-data", type=Path,
                        default=Path("source_videos"))
    parser.add_argument("--output-root", type=Path,
                        default=ROLESTVG_ROOT / "core")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Documented fixed non-date seed used for reproducible window placement",
    )
    parser.add_argument(
        "--contexts", nargs="+", choices=CONTEXTS, default=list(CONTEXTS),
        help="Generate only the selected settings (default: all settings)",
    )
    parser.add_argument("--match-tolerance", type=float, default=2.0)
    parser.add_argument("--safety-margin-sec", type=float, default=10.0)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manifests-only", action="store_true",
        help="Write complete retained manifests without encoding context videos",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--existing-context-video-root", type=Path,
        help=("In manifests-only mode, reference already materialized context "
              "videos below this core root"),
    )
    parser.add_argument("--timestamp-corrections", type=Path,
                        help="Required sidecar for corrected generation; no fallback")
    parser.add_argument("--allow-probable-corrections", action="store_true",
                        help="Use probable corrections (disabled by default)")
    parser.add_argument("--source-boundary-tolerance-sec", type=float, default=0.0,
                        help="Clamp audited verified negative starts within this tolerance")
    parser.add_argument("--source-boundary-audit-pass-manifest", type=Path,
                        help="JSONL manual-audit decisions required for boundary clamps")
    parser.add_argument("--policy-version", type=int, choices=SUPPORTED_POLICY_VERSIONS,
                        default=7)
    parser.add_argument(
        "--manual-exclusions", type=Path,
        help="Reviewer-backed JSONL exclusions applied before target/inventory discovery",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = Roots(
        args.trimmed_clips.resolve(), args.role_annotations.resolve(),
        args.trim_annotations.resolve(), args.soccernet_data.resolve(),
    )
    validate_roots(roots)
    event_index = build_event_index(roots)
    index_path = args.output_root / "event_index.jsonl"
    event_index.write_jsonl(index_path)
    index_stats_path = args.output_root / "event_index_stats.json"
    coverage = source_coverage(roots, event_index, args.match_tolerance)
    index_stats = {
        **coverage,
        "sequence_event_type_distribution": event_index.type_counts(),
        "counts_by_game_half_sequence_event_type":
            event_index.counts_by_game_half_type(),
    }
    manual_exclusions = {}
    manual_exclusion_provenance = None
    if args.manual_exclusions is not None:
        manual_exclusions, manual_exclusion_provenance = load_manual_exclusions(
            args.manual_exclusions.resolve()
        )
        index_stats["manual_exclusions"] = manual_exclusion_provenance
    targets, trim_inventory, removed = discover_trim_inventory(
        roots, event_index, args.match_tolerance, manual_exclusions
    )
    correction_provenance = None
    if args.timestamp_corrections is not None:
        targets, trim_inventory, removed, correction_provenance = apply_corrections(
            targets, trim_inventory, removed,
            args.timestamp_corrections.resolve(),
            args.allow_probable_corrections,
            args.source_boundary_tolerance_sec,
            (args.source_boundary_audit_pass_manifest.resolve()
             if args.source_boundary_audit_pass_manifest else None),
            2,
            args.ffprobe,
        )
        index_stats["timestamp_correction"] = correction_provenance
    index_stats_path.write_text(
        json.dumps(index_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = generate_all(
        targets, trim_inventory, removed, event_index, args.output_root,
        args.seed, args.dry_run, args.overwrite, args.ffprobe,
        args.safety_margin_sec,
        args.manifests_only,
        correction_provenance,
        (args.existing_context_video_root.resolve()
         if args.existing_context_video_root else None),
        args.policy_version,
        manual_exclusion_provenance,
        tuple(args.contexts),
    )
    print(json.dumps({
        "event_index": str(index_path),
        "event_index_stats": str(index_stats_path),
        "matched_targets": len(targets),
        "matched_trim_inventory": len(trim_inventory),
        "unusable_targets": len(removed),
        "source_coverage": coverage,
        "timestamp_correction": correction_provenance,
        "datasets": summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
