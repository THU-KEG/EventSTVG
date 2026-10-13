#!/usr/bin/env python3
"""Run Role-STVG manifests through OpenRouter using ordered video frames."""
from __future__ import annotations

import argparse, json, os, re, sys
import stat
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT / "inference_openrouter" / "configs" / "openrouter.env"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from inference_platform import core
try:
    from .provider import OpenRouterFrameAdapter, OpenRouterNativeVideoAdapter
except ImportError:  # direct ``python inference_openrouter/run.py``
    from inference_openrouter.provider import OpenRouterFrameAdapter, OpenRouterNativeVideoAdapter

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setting", required=True)
    p.add_argument("--model", default="openai/gpt-5.6-sol")
    p.add_argument("--input-mode", choices=("auto", "frames", "native-video"), default="auto",
                   help="Media path; auto uses native-video for Qwen3.5-9B and frames otherwise")
    p.add_argument("--manifest-dir", type=Path, default=ROOT / "prompt")
    p.add_argument("--video-root", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--sample-id", action="append", default=[])
    p.add_argument("--env-file", type=Path, default=None,
                   help="Protected env file (OPENROUTER_API_KEY and OPENROUTER_BASE_URL)")
    p.add_argument("--api-base-url", default=None)
    p.add_argument("--fps", type=float, default=2.0, help="Frames sampled per second")
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--image-detail", choices=("low", "omit"), default="low",
                   help="Set image_url detail=low, or omit detail and use provider default")
    p.add_argument("--provider-order", action="append", default=[],
                   help="Pin OpenRouter provider slug (repeatable); required for formal Qwen runs")
    p.add_argument("--formal-temporal-manifest", type=Path,
                   default=None,
                   help="Absolute window_path manifest used for context_1min runtime resolution")
    args = p.parse_args(argv)
    if args.workers < 1: p.error("--workers must be positive")
    if args.fps <= 0: p.error("--fps must be positive")
    if args.max_frames < 1: p.error("--max-frames must be positive")
    manifests = core.discover_manifests(args.manifest_dir.expanduser())
    if args.setting not in manifests:
        p.error(f"setting {args.setting!r} unavailable; discovered: {sorted(manifests)}")
    key, env_base = None, None
    env_file = (args.env_file or DEFAULT_ENV_FILE).expanduser()
    if args.env_file and not env_file.is_file():
        p.error(f"env file does not exist: {env_file}")
    if env_file.is_file():
        mode = stat.S_IMODE(env_file.stat().st_mode)
        if mode & 0o077:
            p.error(f"{env_file}: must be chmod 600 (current mode {mode:o})")
        try:
            from dotenv import dotenv_values
            values = dotenv_values(env_file)
        except ImportError:
            values = {}
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    name, value = line.split("=", 1)
                    values[name.strip()] = value.strip().strip("\"'")
        key = values.get("OPENROUTER_API_KEY") or values.get("SCHOOL_API_KEY")
        env_base = values.get("OPENROUTER_BASE_URL") or values.get("SCHOOL_BASE_URL")
    key = key or os.environ.get("OPENROUTER_API_KEY")
    if not key: p.error(f"API key is required (set OPENROUTER_API_KEY or use {env_file})")
    base_url = args.api_base_url or env_base or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    root = args.video_root or (Path(os.environ["ROLE_STVG_VIDEO_ROOT"]) if os.environ.get("ROLE_STVG_VIDEO_ROOT") else None)
    if root is None: p.error("--video-root or ROLE_STVG_VIDEO_ROOT is required")
    out = args.output.expanduser().with_suffix(".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out.parent / ".runs" / f"{out.stem}--{re.sub(r'[^A-Za-z0-9_.-]+', '_', args.model)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_path = run_dir / "stop.requested"
    stop_path.unlink(missing_ok=True)
    video_path_map = None
    if args.setting == "context_1min" and args.formal_temporal_manifest is not None:
        temporal_manifest = args.formal_temporal_manifest.expanduser()
        if not temporal_manifest.is_file():
            p.error(f"formal temporal manifest does not exist: {temporal_manifest}")
        video_path_map = {}
        for line in temporal_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id, window_path = item.get("sample_id"), item.get("window_path")
            if isinstance(sample_id, str) and isinstance(window_path, str):
                video_path_map[sample_id] = window_path
        manifest_ids = {r["sample_id"] for r in core.load_manifest(manifests[args.setting])}
        if manifest_ids != set(video_path_map):
            p.error("formal temporal manifest sample IDs do not exactly match frozen context_1min prompt")
        if any(not Path(path).is_file() for path in video_path_map.values()):
            p.error("formal temporal manifest contains missing window_path files")
    config = core.RunConfig(manifest_path=str(manifests[args.setting]), video_root=str(root.expanduser()),
                            output_path=str(out), model=args.model, api_base_url=base_url,
                            workers=args.workers, sample_limit=args.limit, sample_ids=args.sample_id,
                            video_path_map=video_path_map)
    native = args.input_mode == "native-video" or (
        args.input_mode == "auto" and args.model == "qwen/qwen3.5-9b")
    if native:
        adapter = lambda: OpenRouterNativeVideoAdapter(
            api_key=key, base_url=base_url,
            provider_order=args.provider_order or None)
    else:
        adapter = lambda: OpenRouterFrameAdapter(api_key=key, base_url=base_url,
                                                 fps=args.fps, max_frames=args.max_frames,
                                                 image_detail=None if args.image_detail == "omit" else args.image_detail)
    summary = core.run_inference(config, api_key=key, adapter_factory=adapter,
                                 stop_path=stop_path, metadata_path=run_dir / "run.json",
                                 summary_path=run_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary.get("status") == "complete_with_failures" else 0

if __name__ == "__main__":
    raise SystemExit(main())
