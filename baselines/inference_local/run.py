#!/usr/bin/env python3
"""Run frozen Role-STVG manifests through the school OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import sys
from typing import Iterable

from inference_platform.core import (
    PROTOCOL,
    RunConfig,
    RuntimeApiProfile,
    discover_manifests,
    read_jsonl,
    run_inference,
    run_inference_pool,
)
from inference_platform.provider import OpenAICompatibleAdapter, OssSignedHttpsVideoUrl
from inference_local.api_pool import load_api_pool


BENCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = BENCH_ROOT / "prompt"
DEFAULT_ENV_FILE = BENCH_ROOT / "inference_local" / "configs" / "api_shards" / "glm_00.env"
DEFAULT_OUTPUT_DIR = BENCH_ROOT / "soccerrole" / "inference_results_provider_defaults_local"


def safe_name(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return rendered or "model"


def read_sample_ids(paths: Iterable[Path]) -> list[str]:
    values: list[str] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if value in values:
                continue
            values.append(value)
    return values


def unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def resolve_output_path(
    *, output: Path | None, output_dir: Path, model: str, setting: str,
) -> Path:
    if output is None:
        return output_dir.expanduser() / safe_name(model) / f"{setting}.jsonl"
    path = output.expanduser()
    return path if path.suffix == ".jsonl" else path.with_suffix(".jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, help="Setting discovered from the manifest directory")
    parser.add_argument("--model", required=True, help="School API model name or alias")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--video-root", type=Path,
        help="Runtime data root; defaults to ROLE_STVG_VIDEO_ROOT",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output", type=Path,
        help="Result JSONL path; .jsonl is appended when omitted",
    )
    output_group.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--workers", type=int,
        help="Workers for single-API mode (default: 1); pool workers come from YAML",
    )
    parser.add_argument("--limit", type=int, help="Maximum selected samples for a smoke test")
    parser.add_argument("--sample-id", action="append", default=[], help="Select this exact sample ID; repeatable")
    parser.add_argument("--samples-file", action="append", type=Path, default=[], help="File of selected sample IDs")
    parser.add_argument(
        "--rerun-sample", action="append", default=[],
        help="Explicitly rerun this sample even if a successful response is saved; repeatable",
    )
    parser.add_argument(
        "--rerun-samples", action="append", type=Path, default=[],
        help="File of sample IDs to rerun explicitly",
    )
    parser.add_argument(
        "--rerun-failed", action="store_true",
        help="Explicitly rerun every sample whose latest saved attempt is an error",
    )
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument(
        "--env-file", type=Path,
        help=f"Single-API env file (default: {DEFAULT_ENV_FILE})",
    )
    credentials.add_argument(
        "--api-pool", type=Path,
        help="YAML pool referencing protected per-API env files",
    )
    parser.add_argument(
        "--api-base-url", help="Override SCHOOL_BASE_URL; API key remains environment-only",
    )
    parser.add_argument(
        "--video-transport", choices=("base64", "oss_signed_https"), default="base64",
        help="Video transport only; model-facing inference semantics are unchanged",
    )
    parser.add_argument(
        "--signed-video-oss-root",
        help="OSS root corresponding exactly to --video-root (required for signed HTTPS)",
    )
    parser.add_argument(
        "--signed-video-local-root", type=Path,
        help="Resolved local root corresponding to the signed OSS root (defaults to --video-root)",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    scheduler_options: dict | None = None,
) -> int:
    """Run the local CLI through the shared scheduler.

    ``scheduler_options`` exists only for simulated-time integration tests; the
    command-line path uses the shared scheduler's production defaults.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.video_transport == "oss_signed_https" and not args.signed_video_oss_root:
        parser.error("--signed-video-oss-root is required for signed HTTPS transport")

    manifests = discover_manifests(args.manifest_dir.expanduser())
    if args.setting not in manifests:
        parser.error(
            f"setting {args.setting!r} is not available; discovered: {sorted(manifests)}"
        )

    api_key = ""
    pool_profiles = []
    if args.api_pool:
        if args.workers is not None:
            parser.error("--workers cannot be used with --api-pool; set workers per profile")
        if args.api_base_url:
            parser.error("--api-base-url cannot be used with --api-pool; use profile env files")
        try:
            pool_profiles = load_api_pool(args.api_pool, model=args.model)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        base_url = pool_profiles[0].base_url
        workers = sum(profile.workers for profile in pool_profiles)
    else:
        from dotenv import load_dotenv

        env_file = (args.env_file or DEFAULT_ENV_FILE).expanduser()
        load_dotenv(env_file, override=False)
        api_key = os.environ.get("SCHOOL_API_KEY", "")
        base_url = args.api_base_url or os.environ.get("SCHOOL_BASE_URL", "")
        if not api_key or not base_url:
            parser.error(
                f"SCHOOL_API_KEY and SCHOOL_BASE_URL are required (checked {env_file})"
            )
        workers = args.workers or 1
    video_root = args.video_root or (
        Path(os.environ["ROLE_STVG_VIDEO_ROOT"])
        if os.environ.get("ROLE_STVG_VIDEO_ROOT") else None
    )
    if video_root is None:
        parser.error("--video-root or ROLE_STVG_VIDEO_ROOT is required")

    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    output_path = resolve_output_path(
        output=args.output, output_dir=output_dir,
        model=args.model, setting=args.setting,
    ).resolve()
    rerun_ids = unique([
        *args.rerun_sample,
        *read_sample_ids(args.rerun_samples),
    ])
    if args.rerun_failed and output_path.is_file():
        latest = {
            row["sample_id"]: row for row in read_jsonl(output_path)
            if isinstance(row.get("sample_id"), str)
        }
        rerun_ids = unique([
            *rerun_ids,
            *(sample_id for sample_id, row in latest.items() if row.get("status") == "error"),
        ])
    selected_ids = unique([
        *args.sample_id,
        *read_sample_ids(args.samples_file),
    ])
    # A standalone explicit rerun is also a sample selection. Without this,
    # unseen manifest samples would be launched alongside the nominated rerun.
    if rerun_ids and not selected_ids:
        selected_ids = list(rerun_ids)
    run_dir = output_path.parent / ".runs" / f"{output_path.stem}--{safe_name(args.model)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_path = run_dir / "stop.requested"
    stop_path.unlink(missing_ok=True)

    config = RunConfig(
        manifest_path=str(manifests[args.setting]),
        video_root=str(video_root.expanduser()),
        output_path=str(output_path),
        model=args.model,
        api_base_url=base_url,
        workers=workers,
        sample_limit=args.limit,
        sample_ids=selected_ids,
        rerun_sample_ids=rerun_ids,
    )
    shared_scheduler_options = scheduler_options or {}
    video_url_factory = None
    if args.video_transport == "oss_signed_https":
        video_url_factory = OssSignedHttpsVideoUrl(
            local_root=args.signed_video_local_root or video_root,
            oss_root=args.signed_video_oss_root,
        )

    def request_safe_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
        stop_path.touch()
        print("\nSafe stop requested; draining active requests...", file=sys.stderr, flush=True)

    previous_handler = signal.signal(signal.SIGINT, request_safe_stop)
    print(f"Inference protocol: {PROTOCOL}")
    print("Optional model-facing parameters sent: None")
    print(f"Manifest: {manifests[args.setting]}")
    print(f"Output: {output_path}")
    if pool_profiles:
        print(
            "API pool: "
            + ", ".join(f"{profile.name}({profile.workers})" for profile in pool_profiles)
        )
    if rerun_ids:
        print(f"Explicit rerun samples: {len(rerun_ids)}")
    try:
        if pool_profiles:
            runtime_profiles = [
                RuntimeApiProfile(
                    name=profile.name,
                    api_key=profile.api_key,
                    base_url=profile.base_url,
                    workers=profile.workers,
                    adapter_factory=(
                        lambda key=profile.api_key, url=profile.base_url:
                        OpenAICompatibleAdapter(
                            api_key=key, base_url=url,
                            video_url_factory=video_url_factory,
                        )
                    ),
                )
                for profile in pool_profiles
            ]
            summary = run_inference_pool(
                config, api_profiles=runtime_profiles,
                stop_path=stop_path, metadata_path=run_dir / "run.json",
                summary_path=run_dir / "summary.json",
                **shared_scheduler_options,
            )
        else:
            summary = run_inference(
                config,
                api_key=api_key,
                adapter_factory=lambda: OpenAICompatibleAdapter(
                    api_key=api_key, base_url=base_url,
                    video_url_factory=video_url_factory,
                ),
                stop_path=stop_path,
                metadata_path=run_dir / "run.json",
                summary_path=run_dir / "summary.json",
                **shared_scheduler_options,
            )
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary.get("status") == "complete_with_failures" else 0


if __name__ == "__main__":
    raise SystemExit(main())
