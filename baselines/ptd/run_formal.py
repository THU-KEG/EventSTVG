#!/usr/bin/env python3
"""Resume-safe, single-worker formal PTD-64f Role-STVG inference runner."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import decord


ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "policy"
PTD_ROOT = Path(os.environ.get("PTD_CODE_ROOT", "models/ParallelTubeDecoding"))
MODEL_PATH = Path(os.environ.get("PTD_MODEL_PATH", "models/ParallelTubeDecoding-Qwen3-VL-4B"))
MAX_FRAMES = 64
MIN_PIXELS = 131072
MAX_PIXELS = 786432

sys.path.insert(0, str(PTD_ROOT / "src"))
data_utils_spec = importlib.util.spec_from_file_location(
    "ptd_official_data_utils", PTD_ROOT / "src" / "dataset" / "data_utils.py"
)
data_utils = importlib.util.module_from_spec(data_utils_spec)
data_utils_spec.loader.exec_module(data_utils)
patch_processor_with_time_tokens = data_utils.patch_processor_with_time_tokens
patch_qwen3_video_processor = data_utils.patch_qwen3_video_processor
from model.ptd_generation import generate_ptd  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
import qwen_vl_utils.vision_process as qwen_vision_process  # noqa: E402
from utils import disable_torch_init, get_model_name_from_path, load_pretrained_model  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_event(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_frozen_files() -> dict[str, str]:
    sums_path = POLICY_DIR / "SHA256SUMS"
    if not sums_path.is_file():
        raise RuntimeError(f"Missing frozen hash file: {sums_path}")
    verified = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        expected, label = line.split("  ", 1)
        path = Path(label) if label.startswith("/") else ROOT / label
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen-file hash mismatch for {path}: expected {expected}, got {actual}"
            )
        verified[label] = actual
    return verified


def read_manifest(path: Path) -> list[dict]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            sample_id = record["sample_id"]
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id at {path}:{line_number}: {sample_id}")
            seen.add(sample_id)
            rows.append(record)
    return rows


def parse_record(record: dict) -> tuple[str, list[tuple[str, str]]]:
    prompt = record["prompt"]
    section = prompt.split("## EVENT DEFINITION\n\n", 1)
    if len(section) != 2:
        raise ValueError(f"Missing event definition: {record['sample_id']}")
    event_match = re.match(r"\*\*([^*]+)\*\*:", section[1])
    if not event_match:
        raise ValueError(f"Cannot parse event: {record['sample_id']}")
    roles = re.findall(r"^- `([^`]+)`:\s*(.+)$", prompt, flags=re.MULTILINE)
    if not roles:
        raise ValueError(f"Cannot parse roles: {record['sample_id']}")
    return event_match.group(1), [(key, value.strip()) for key, value in roles]


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def video_path_for(record: dict, setting: str, protocol: dict) -> Path:
    if setting == "oracle":
        return Path(protocol["inputs"]["oracle_video_root"]) / record["video_path"]
    return Path(protocol["inputs"]["context_1min_video_root"]) / (
        record["sample_id"] + ".mp4"
    )


def inspect_video(path: Path) -> tuple[decord.VideoReader, dict]:
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    count = len(reader)
    fps = float(reader.get_avg_fps())
    if count <= 0 or fps <= 0:
        raise ValueError(f"Invalid video metadata: frames={count}, fps={fps}")
    height, width = map(int, reader[0].shape[:2])
    return reader, {
        "source_frame_count": count,
        "source_fps": fps,
        "source_duration": count / fps,
        "source_resolution_hw": [height, width],
    }


def video_spec(path: Path, nframes: int) -> dict:
    return {
        "type": "video",
        "video": str(path),
        "nframes": nframes,
        "temporal_patch_size": 1,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
    }


def prepare_shared_visual(path: Path, nframes: int, placeholder_prompt: str):
    spec = video_spec(path, nframes)
    messages = [
        {
            "role": "user",
            "content": [spec, {"type": "text", "text": placeholder_prompt}],
        }
    ]
    qwen_vision_process.FRAME_FACTOR = 1
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )
    tensors, metadata = zip(*video_inputs)
    tensors, metadata = list(tensors), list(metadata)
    indices = [int(value) for value in metadata[0]["frames_indices"]]
    if len(indices) != nframes or len(set(indices)) != nframes:
        raise ValueError(
            f"Sampling mismatch: requested={nframes}, returned={len(indices)}, "
            f"distinct={len(set(indices))}"
        )
    if indices != sorted(indices) or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("Selected source indices are not strictly increasing")
    total = int(metadata[0]["total_num_frames"])
    if indices[0] != 0 or indices[-1] != total - 1:
        raise ValueError(
            f"Uniform sampling does not cover full extent: {indices[0]}..{indices[-1]} "
            f"of 0..{total-1}"
        )
    return {
        "spec": spec,
        "image_inputs": image_inputs,
        "video_tensors": tensors,
        "video_metadata": metadata,
        "video_kwargs": video_kwargs,
        "indices": indices,
        "timestamps": [value / float(metadata[0]["fps"]) for value in indices],
        "decoded_tensor_shape": list(tensors[0].shape),
    }


def prepare_role_inputs(processor, shared: dict, model_prompt: str, nframes: int):
    messages = [
        {
            "role": "user",
            "content": [shared["spec"], {"type": "text", "text": model_prompt}],
        }
    ]
    rendered = processor.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True
    )[0]
    inputs = processor(
        text=[rendered],
        images=shared["image_inputs"],
        videos=shared["video_tensors"],
        video_metadata=shared["video_metadata"],
        **shared["video_kwargs"],
        do_resize=False,
        return_tensors="pt",
    )
    grid = inputs["video_grid_thw"]
    entered = int(grid[0, 0].item())
    if entered != nframes:
        raise ValueError(f"Processor frame mismatch: {entered} != {nframes}")
    input_ids = inputs["input_ids"][0]
    for number in range(1, nframes + 1):
        token_id = processor.tokenizer.convert_tokens_to_ids(f"<t{number}>")
        if int((input_ids == token_id).sum().item()) != 1:
            raise ValueError(f"Time token <t{number}> was not inserted exactly once")
    return inputs.to("cuda:0"), {
        "rendered_model_facing_prompt": rendered,
        "processor_input_shapes": {
            key: list(value.shape)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        },
        "video_grid_thw": grid.tolist(),
        "final_model_frame_count": entered,
        "model_resolution_hw": [
            int(grid[0, 1].item()) * 16,
            int(grid[0, 2].item()) * 16,
        ],
        "inserted_time_token_count": nframes,
    }


BOX_PATTERN = re.compile(
    r"<t(\d+)>\s*<\|box_start\|>\s*"
    r"<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<\|box_end\|>"
)


def parse_native(raw: str) -> dict:
    time_tokens = [int(value) for value in re.findall(r"<t(\d+)>", raw)]
    object_refs = re.findall(
        r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>", raw, re.DOTALL
    )
    segment_match = re.search(
        r"<\|time_start\|>\s*<t(\d+)>\s*<t(\d+)>\s*<\|time_end\|>", raw
    )
    segment = (
        [int(segment_match.group(1)), int(segment_match.group(2))]
        if segment_match
        else None
    )
    boxes = [
        {
            "time_token": int(t),
            "bbox_tokens": [int(x1), int(y1), int(x2), int(y2)],
        }
        for t, x1, y1, x2, y2 in BOX_PATTERN.findall(raw)
    ]
    coherent = bool(segment and boxes)
    if coherent:
        start, end = segment
        box_times = [item["time_token"] for item in boxes]
        coherent = (
            1 <= start <= end <= MAX_FRAMES
            and box_times == sorted(box_times)
            and all(start <= value <= end for value in box_times)
            and all(
                0 <= coordinate <= 1000
                for item in boxes
                for coordinate in item["bbox_tokens"]
            )
        )
    stripped = re.sub(r"(?:<\|im_end\|>|\s)+", "", raw)
    if coherent:
        status = "OK"
    elif not stripped:
        status = "EMPTY"
    else:
        status = "MALFORMED"
    return {
        "status": status,
        "parsed_object_references": object_refs,
        "parsed_native_time_tokens": time_tokens,
        "parsed_native_time_interval": segment,
        "parsed_native_bbox_tokens": boxes,
        "predicted_bbox_count": len(boxes),
    }


def record_id(setting: str, sample_id: str, role_key: str) -> str:
    return sha256_bytes(f"{setting}\0{sample_id}\0{role_key}".encode())[:32]


def write_role_record(output_root: Path, record: dict) -> Path:
    path = output_root / "records" / f"{record['record_id']}.json"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite formal record: {path}")
    raw_path = output_root / "raw_native_outputs" / f"{record['record_id']}.txt"
    atomic_text(raw_path, record.get("raw_native_output", ""))
    record["raw_native_output_path"] = str(raw_path)
    atomic_json(path, record)
    return path


def existing_records(output_root: Path) -> dict[tuple[str, str, str], dict]:
    found = {}
    for path in sorted((output_root / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["setting"], record["sample_id"], record["role_key"])
        if key in found:
            raise RuntimeError(f"Duplicate formal record key {key}: {path}")
        expected_id = record_id(*key)
        if record["record_id"] != expected_id or path.stem != expected_id:
            raise RuntimeError(f"Record identity mismatch: {path}")
        found[key] = record
    return found


def rebuild_results(output_root: Path) -> dict[tuple[str, str, str], dict]:
    found = existing_records(output_root)
    ordered = sorted(found.values(), key=lambda row: (row["manifest_index"], row["role_index"]))
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered)
    atomic_text(output_root / "results.jsonl", text)
    return found


def progress(output_root: Path, expected: int, current: dict | None = None) -> dict:
    found = existing_records(output_root)
    counts = Counter(record["status"] for record in found.values())
    value = {
        "updated_at_utc": utc_now(),
        "expected_role_calls": expected,
        "committed_role_calls": len(found),
        "remaining_role_calls": max(0, expected - len(found)),
        "status_counts": dict(sorted(counts.items())),
        "current": current,
    }
    atomic_json(output_root / "progress.json", value)
    return value


def run_role(
    model,
    processor,
    *,
    output_root: Path,
    policy_id: str,
    setting: str,
    manifest_index: int,
    role_index: int,
    record: dict,
    event_type: str,
    role_key: str,
    definition: str,
    template: dict,
    video_path: Path,
    video_meta: dict,
    shared: dict,
    shared_decode_sec: float,
    nframes: int,
) -> dict:
    rid = record_id(setting, record["sample_id"], role_key)
    result = {
        "record_id": rid,
        "protocol_id": policy_id,
        "committed_at_utc": None,
        "sample_id": record["sample_id"],
        "setting": setting,
        "manifest_index": manifest_index,
        "event_type": event_type,
        "role_index": role_index,
        "role_key": role_key,
        "frozen_role_definition": definition,
        "ptd_query": template["ptd_referring_query"],
        "ptd_model_prompt": template["ptd_model_prompt"],
        "source_video": str(video_path),
        **video_meta,
        "input_frame_count": nframes,
        "selected_source_frame_indices": shared["indices"],
        "selected_timestamps": shared["timestamps"],
        "decoded_frame_tensor_shape": shared["decoded_tensor_shape"],
        "shared_visual_decode_sec": shared_decode_sec,
        "raw_native_output": "",
        "parsed_native_time_tokens": [],
        "parsed_native_bbox_tokens": [],
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        inputs, prep = prepare_role_inputs(
            processor, shared, template["ptd_model_prompt"], nframes
        )
        result.update(prep)
        generation_started = time.perf_counter()
        completion, generation_result = generate_ptd(
            model,
            processor.tokenizer,
            dict(inputs),
            max_new_tokens=1024,
            max_time_tokens=nframes,
            temperature=0.0,
            top_p=None,
            top_k=None,
            generation_format="spatio_temporal_grounding",
            ptd_attn_implementation="sdpa",
            measure_decode=True,
        )
        torch.cuda.synchronize()
        raw = processor.batch_decode(
            completion,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        result["raw_native_output"] = raw
        result.update(parse_native(raw))
        result["generation_sec"] = time.perf_counter() - generation_started
        result["native_generation_result"] = jsonable(generation_result.__dict__)
    except torch.cuda.OutOfMemoryError as exc:
        result.update(
            {
                "status": "CUDA_OOM",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    except Exception as exc:
        status = "CUDA_OOM" if "CUDA out of memory" in str(exc) else "REQUEST_FAILURE"
        result.update(
            {"status": status, "error": repr(exc), "traceback": traceback.format_exc()}
        )
    finally:
        result["runtime_sec"] = time.perf_counter() - started
        result["peak_vram_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        result["peak_reserved_vram_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
        result["committed_at_utc"] = utc_now()
    write_role_record(output_root, result)
    return result


def failure_record(
    *,
    output_root: Path,
    policy_id: str,
    setting: str,
    manifest_index: int,
    role_index: int,
    record: dict,
    event_type: str,
    role_key: str,
    definition: str,
    template: dict,
    video_path: Path,
    exc: Exception,
) -> dict:
    result = {
        "record_id": record_id(setting, record["sample_id"], role_key),
        "protocol_id": policy_id,
        "committed_at_utc": utc_now(),
        "sample_id": record["sample_id"],
        "setting": setting,
        "manifest_index": manifest_index,
        "event_type": event_type,
        "role_index": role_index,
        "role_key": role_key,
        "frozen_role_definition": definition,
        "ptd_query": template["ptd_referring_query"],
        "ptd_model_prompt": template["ptd_model_prompt"],
        "source_video": str(video_path),
        "source_duration": None,
        "input_frame_count": None,
        "selected_source_frame_indices": [],
        "selected_timestamps": [],
        "raw_native_output": "",
        "parsed_native_time_tokens": [],
        "parsed_native_bbox_tokens": [],
        "runtime_sec": 0.0,
        "peak_vram_gib": 0.0,
        "status": "REQUEST_FAILURE",
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    write_role_record(output_root, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--setting", choices=("oracle", "context_1min"))
    group.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--video-base-dir", type=Path, required=True, help="Parent of oracle/ and context_1min/ video directories")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    lock_handle = (ROOT / "single_worker.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("Another PTD worker is active; worker_count must remain 1") from exc

    verified = verify_frozen_files()
    protocol = json.loads((POLICY_DIR / "protocol.json").read_text(encoding="utf-8"))
    role_policy = json.loads(
        (POLICY_DIR / "role_query_templates.json").read_text(encoding="utf-8")
    )
    if protocol["state"] != "FROZEN" or protocol["execution"]["worker_count"] != 1:
        raise RuntimeError("Formal policy is not frozen for exactly one worker")
    expected_allocator = protocol["execution"]["cuda_allocator"]
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != expected_allocator:
        raise RuntimeError(
            "CUDA allocator policy mismatch: expected "
            f"{expected_allocator!r}, got "
            f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r}"
        )
    template_map = {
        (item["event_type"], item["role_key"]): item
        for item in role_policy["templates"]
    }
    protocol["inputs"]["oracle_video_root"] = str(args.video_base_dir / "oracle")
    protocol["inputs"]["context_1min_video_root"] = str(args.video_base_dir / "context_1min")
    manifest_paths = {
        "oracle": args.manifest_dir / "oracle.jsonl",
        "context_1min": args.manifest_dir / "context_1min.jsonl",
    }
    all_rows = {key: read_manifest(path) for key, path in manifest_paths.items()}

    if args.dry_run:
        output_root = args.output_root / "dry_run"
        cases = protocol["dry_run_cases"]
        work = []
        for case_index, case in enumerate(cases):
            rows = all_rows[case["setting"]]
            matches = [
                (index, row)
                for index, row in enumerate(rows)
                if row["sample_id"] == case["sample_id"]
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Dry-run sample resolution failure: {case}")
            manifest_index, row = matches[0]
            work.append((case["setting"], manifest_index, row))
    else:
        output_root = args.output_root / args.setting
        work = [
            (args.setting, manifest_index, row)
            for manifest_index, row in enumerate(all_rows[args.setting])
        ]
    (output_root / "records").mkdir(parents=True, exist_ok=True)
    (output_root / "raw_native_outputs").mkdir(parents=True, exist_ok=True)

    expected = sum(len(parse_record(row)[1]) for _, _, row in work)
    committed = rebuild_results(output_root)
    progress(output_root, expected)
    append_event(
        output_root / "events.jsonl",
        {
            "timestamp_utc": utc_now(),
            "event": "RUN_STARTED",
            "mode": "dry_run" if args.dry_run else args.setting,
            "worker_count": 1,
            "verified_hashes": verified,
            "existing_records": len(committed),
            "expected_role_calls": expected,
        },
    )

    disable_torch_init()
    load_started = time.perf_counter()
    processor, model = load_pretrained_model(
        model_base=None,
        model_path=str(MODEL_PATH),
        model_name=get_model_name_from_path(str(MODEL_PATH)),
        device_map="cuda:0",
        device="cuda",
        load_4bit=False,
        load_8bit=False,
        use_flash_attn=False,
    )
    model.eval()
    patch_qwen3_video_processor(processor)
    patch_processor_with_time_tokens(processor, max_time_tokens=100)
    torch.cuda.synchronize()
    for token in ("<t1>", "<t64>", "<t100>"):
        if processor.tokenizer.convert_tokens_to_ids(token) is None:
            raise RuntimeError(f"Missing checkpoint token {token}")
    append_event(
        output_root / "events.jsonl",
        {
            "timestamp_utc": utc_now(),
            "event": "MODEL_LOADED",
            "seconds": time.perf_counter() - load_started,
            "dtype": str(model.dtype),
        },
    )

    new_since_rebuild = 0
    for setting, manifest_index, source_record in work:
        event_type, roles = parse_record(source_record)
        video_path = video_path_for(source_record, setting, protocol)
        pending = [
            (role_index, role_key, definition)
            for role_index, (role_key, definition) in enumerate(roles)
            if (setting, source_record["sample_id"], role_key) not in committed
        ]
        if not pending:
            continue
        current = {
            "setting": setting,
            "manifest_index": manifest_index,
            "sample_id": source_record["sample_id"],
            "event_type": event_type,
        }
        progress(output_root, expected, current)
        try:
            visual_started = time.perf_counter()
            _, video_meta = inspect_video(video_path)
            nframes = (
                min(MAX_FRAMES, video_meta["source_frame_count"])
                if setting == "oracle"
                else MAX_FRAMES
            )
            placeholder_template = template_map[(event_type, pending[0][1])]
            shared = prepare_shared_visual(
                video_path, nframes, placeholder_template["ptd_model_prompt"]
            )
            shared_decode_sec = time.perf_counter() - visual_started
        except Exception as exc:
            for role_index, role_key, definition in pending:
                template = template_map[(event_type, role_key)]
                result = failure_record(
                    output_root=output_root,
                    policy_id=protocol["protocol_id"],
                    setting=setting,
                    manifest_index=manifest_index,
                    role_index=role_index,
                    record=source_record,
                    event_type=event_type,
                    role_key=role_key,
                    definition=definition,
                    template=template,
                    video_path=video_path,
                    exc=exc,
                )
                committed[(setting, source_record["sample_id"], role_key)] = result
                print(json.dumps({"record_id": result["record_id"], "status": result["status"], **current, "role_key": role_key}), flush=True)
                new_since_rebuild += 1
            progress(output_root, expected, current)
            continue
        for role_index, role_key, definition in pending:
            template = template_map.get((event_type, role_key))
            if template is None or template["frozen_role_definition"] != definition:
                raise RuntimeError(
                    f"Frozen template mismatch: {(event_type, role_key, definition)}"
                )
            result = run_role(
                model,
                processor,
                output_root=output_root,
                policy_id=protocol["protocol_id"],
                setting=setting,
                manifest_index=manifest_index,
                role_index=role_index,
                record=source_record,
                event_type=event_type,
                role_key=role_key,
                definition=definition,
                template=template,
                video_path=video_path,
                video_meta=video_meta,
                shared=shared,
                shared_decode_sec=shared_decode_sec,
                nframes=nframes,
            )
            key = (setting, source_record["sample_id"], role_key)
            committed[key] = result
            append_event(
                output_root / "events.jsonl",
                {
                    "timestamp_utc": utc_now(),
                    "event": "ROLE_COMMITTED",
                    "record_id": result["record_id"],
                    "sample_id": source_record["sample_id"],
                    "role_key": role_key,
                    "status": result["status"],
                    "runtime_sec": result["runtime_sec"],
                },
            )
            print(
                json.dumps(
                    {
                        "record_id": result["record_id"],
                        "status": result["status"],
                        **current,
                        "role_key": role_key,
                        "frames": nframes,
                        "runtime_sec": result["runtime_sec"],
                        "peak_vram_gib": result["peak_vram_gib"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            new_since_rebuild += 1
            progress(output_root, expected, current)
            if new_since_rebuild >= 25:
                rebuild_results(output_root)
                new_since_rebuild = 0

    rebuild_results(output_root)
    final_progress = progress(output_root, expected)
    append_event(
        output_root / "events.jsonl",
        {
            "timestamp_utc": utc_now(),
            "event": "RUN_COMPLETED",
            "progress": final_progress,
        },
    )
    print(json.dumps(final_progress, ensure_ascii=False), flush=True)
    if args.dry_run:
        records = existing_records(output_root)
        failures = [row for row in records.values() if row["status"] != "OK"]
        if failures:
            raise SystemExit(
                "Dry run failed: "
                + json.dumps(
                    [
                        {
                            "sample_id": row["sample_id"],
                            "role_key": row["role_key"],
                            "status": row["status"],
                        }
                        for row in failures
                    ],
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
