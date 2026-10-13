#!/usr/bin/env python3
"""Materialize policy-v8 context manifests at a controlled H.264 bitrate.

The runner is resumable: validated outputs are skipped, incomplete encodes use
temporary files, and each completed output is appended to a JSONL event log.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


CONTEXTS = ("context_1min", "context_3min", "context_5min", "context_10min")
LOG_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-manifest-root", type=Path,
        help="Process only sample IDs absent from this baseline manifest root",
    )
    parser.add_argument(
        "--expected-selected-per-context", type=int,
        help="Fail unless each context selects exactly this many samples",
    )
    parser.add_argument("--contexts", nargs="+", choices=CONTEXTS,
                        default=list(CONTEXTS))
    parser.add_argument("--video-bitrate", default="2500k")
    parser.add_argument("--audio-bitrate", default="128k")
    parser.add_argument("--maxrate", default="3000k")
    parser.add_argument("--bufsize", default="5000k")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--minimum-free-gb", type=float, default=40.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def probe(path: Path) -> dict:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate",
        "-show_entries", "format=duration,bit_rate", "-of", "json", str(path),
    ], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise RuntimeError("no video stream")
    return data


def valid_output(path: Path, expected_duration: float) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        data = probe(path)
        stream = data["streams"][0]
        duration = float(data["format"]["duration"])
        return (
            stream.get("codec_name") == "h264"
            and int(stream.get("width", 0)) == 1280
            and int(stream.get("height", 0)) == 720
            and stream.get("avg_frame_rate") == "25/1"
            and abs(duration - expected_duration) <= 0.25
        )
    except Exception:
        return False


def append_event(log_path: Path, payload: dict) -> None:
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload}
    with LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def load_jobs(args: argparse.Namespace) -> list[dict]:
    jobs = []
    seen = set()
    selected_by_context = {}
    for context in args.contexts:
        manifest_path = args.manifest_root / context / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        retained = manifest["retained"]
        selected = retained
        if args.baseline_manifest_root:
            baseline_path = args.baseline_manifest_root / context / "manifest.json"
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["retained"]
            baseline_ids = {item["sample_id"] for item in baseline}
            selected = [item for item in retained if item["sample_id"] not in baseline_ids]
        selected_by_context[context] = len(selected)
        if (args.expected_selected_per_context is not None
                and len(selected) != args.expected_selected_per_context):
            raise RuntimeError(
                f"Expected {args.expected_selected_per_context} selected samples in "
                f"{manifest_path}, got {len(selected)}"
            )
        for item in selected:
            sample_id = item["sample_id"]
            output = args.output_root / context / "videos" / f"{sample_id}.mp4"
            key = str(output)
            if key in seen:
                raise RuntimeError(f"Duplicate output path: {output}")
            seen.add(key)
            duration = float(item["context_duration"])
            start = float(item["context_start"])
            end = float(item["context_end"])
            if abs((end - start) - duration) > 1e-5:
                raise RuntimeError(f"Inconsistent context interval for {sample_id}")
            source = Path(item["source_half_video"])
            if not source.is_file():
                raise RuntimeError(f"Missing source video: {source}")
            jobs.append({
                "context": context, "sample_id": sample_id, "source": source,
                "output": output, "start": start, "duration": duration,
            })
    if args.baseline_manifest_root:
        selected_sets = []
        for context in args.contexts:
            current = json.loads(
                (args.manifest_root / context / "manifest.json").read_text(encoding="utf-8")
            )["retained"]
            baseline = json.loads(
                (args.baseline_manifest_root / context / "manifest.json").read_text(encoding="utf-8")
            )["retained"]
            baseline_ids = {item["sample_id"] for item in baseline}
            selected_sets.append({item["sample_id"] for item in current} - baseline_ids)
        if selected_sets and any(ids != selected_sets[0] for ids in selected_sets[1:]):
            raise RuntimeError("Selected sample-ID delta differs across contexts")
    return jobs


def encode(job: dict, args: argparse.Namespace, log_path: Path) -> str:
    output = job["output"]
    if valid_output(output, job["duration"]):
        return "skipped"
    free_gb = shutil.disk_usage(args.output_root).free / 1e9
    if free_gb < args.minimum_free_gb:
        raise RuntimeError(
            f"Disk guard triggered: {free_gb:.2f} GB free, "
            f"minimum is {args.minimum_free_gb:.2f} GB"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".encoding.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-ss", f"{job['start']:.6f}", "-i", str(job["source"]),
        "-t", f"{job['duration']:.6f}",
        "-vf", "scale=1280:720,fps=25", "-c:v", "libx264",
        "-preset", args.preset, "-b:v", args.video_bitrate,
        "-maxrate", args.maxrate, "-bufsize", args.bufsize,
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", args.audio_bitrate,
        "-movflags", "+faststart", "-avoid_negative_ts", "make_zero",
        str(temporary),
    ]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")
    if not valid_output(temporary, job["duration"]):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("encoded output failed validation")
    temporary.replace(output)
    append_event(log_path, {
        "event": "completed", "context": job["context"],
        "sample_id": job["sample_id"], "output": str(output),
        "size_bytes": output.stat().st_size,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    return "completed"


def main() -> None:
    args = parse_args()
    args.manifest_root = args.manifest_root.resolve()
    if args.baseline_manifest_root:
        args.baseline_manifest_root = args.baseline_manifest_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(args)
    estimated_seconds = sum(j["duration"] for j in jobs)
    estimated_bytes = estimated_seconds * (2_500_000 + 128_000) / 8
    plan = {
        "contexts": args.contexts, "jobs": len(jobs),
        "total_duration_hours": round(estimated_seconds / 3600, 3),
        "estimated_output_gb": round(estimated_bytes / 1e9, 3),
        "free_gb": round(shutil.disk_usage(args.output_root).free / 1e9, 3),
        "settings": {
            "resolution": "1280x720", "fps": 25, "codec": "h264",
            "video_bitrate": args.video_bitrate,
            "audio_bitrate": args.audio_bitrate, "preset": args.preset,
        },
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    log_path = args.output_root / "low_bitrate_generation_events.jsonl"
    counts = {"completed": 0, "skipped": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_job = {pool.submit(encode, job, args, log_path): job for job in jobs}
        for index, future in enumerate(concurrent.futures.as_completed(future_to_job), 1):
            job = future_to_job[future]
            try:
                status = future.result()
            except Exception as exc:
                append_event(log_path, {
                    "event": "failed", "context": job["context"],
                    "sample_id": job["sample_id"], "error": str(exc),
                })
                raise
            counts[status] += 1
            if index % 25 == 0 or index == len(jobs):
                print(json.dumps({"progress": index, "total": len(jobs), **counts}),
                      flush=True)
    print(json.dumps({"status": "complete", **counts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
