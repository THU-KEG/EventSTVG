from __future__ import annotations
import argparse, json, os, tempfile, time, traceback
from pathlib import Path
import torch
from .registry import get_adapter
try:
    from parser_policy import extract_prediction
except ImportError:
    extract_prediction = None

BENCH = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BENCH / "prompt" / "oracle.jsonl"
DEFAULT_OUTPUT = BENCH / "outputs" / "Qwen3.5-9B" / "oracle.jsonl"

def _rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def _failed(row: dict) -> bool:
    return str(row.get("status", "")).endswith("failure") or row.get("status") == "error"

def _rewrite_without(path: Path, sample_ids: set[str]) -> None:
    """Atomically remove retry targets while preserving every other record."""
    rows = _rows(path) if path.exists() else []
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                if row.get("sample_id") not in sample_ids:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def run(args) -> list[dict]:
    rows = _rows(args.manifest)
    selected = [r for r in rows if not args.sample_id or r["sample_id"] in args.sample_id]
    if args.limit: selected = selected[:args.limit]
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    existing = _rows(output_path) if output_path.exists() else []
    by_id = {r.get("sample_id"): r for r in existing if r.get("sample_id")}
    if args.retry_failures:
        retry_ids = {r["sample_id"] for r in selected if _failed(by_id.get(r["sample_id"], {}))}
        _rewrite_without(output_path, retry_ids)
        existing = [r for r in existing if r.get("sample_id") not in retry_ids]
        by_id = {r.get("sample_id"): r for r in existing if r.get("sample_id")}
        selected = [r for r in selected if r["sample_id"] in retry_ids]
    else:
        selected = [r for r in selected if r["sample_id"] not in by_id]
    adapter = get_adapter(args.model, model_path=args.model_path)
    load_error = None
    try: adapter.load()
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    output = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if load_error:
        return [{"model": args.model, "status": "model_load_failure", "error": load_error}]
    try:
        with output_path.open("a", encoding="utf-8") as output_stream:
          for row in selected:
            started = time.perf_counter(); rec = {"model": args.model, "sample_id": row["sample_id"],
                "setting": row["setting"], "raw_response": None, "status": "success", "error": None,
                "load_seconds": adapter.load_seconds,
                "generated_token_count": None,
                "max_new_tokens": int(getattr(adapter, "max_new_tokens", 0)) or None,
                "hit_max_new_tokens": False}
            try:
                video = Path(args.video_root) / row["video_path"]
                inp = adapter.prepare_inputs(video, row["prompt"])
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                raw = adapter.generate(inp); rec["raw_response"] = adapter.decode(raw, inp)
                generated_count = int(raw.shape[-1] - inp["prompt_tokens"])
                budget = int(getattr(adapter, "max_new_tokens", 0))
                rec["generated_token_count"] = generated_count
                rec["max_new_tokens"] = budget
                rec["hit_max_new_tokens"] = bool(budget and generated_count >= budget)
                if extract_prediction is not None:
                    parsed = extract_prediction(rec["raw_response"], row["setting"])
                    rec["parser_status"] = parsed.status
                    rec["parser_ok"] = parsed.parse_ok
                rec["inference_seconds"] = time.perf_counter() - started
                rec["peak_gpu_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            except Exception as exc:
                rec.update(status="preprocessing_failure" if "decode" in str(exc).lower() or "video" in str(exc).lower() else "generation_failure",
                           error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                           inference_seconds=time.perf_counter()-started)
            output.append(rec)
            output_stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
            output_stream.flush()
    finally: adapter.unload()
    return output

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model", default="qwen3.5-9b"); p.add_argument("--model-path"); p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST); p.add_argument("--video-root", required=True); p.add_argument("--sample-id", action="append", default=[]); p.add_argument("--limit", type=int); p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); p.add_argument("--retry-failures", action="store_true")
    a=p.parse_args(); result=run(a); print("\n".join(json.dumps(x, ensure_ascii=False) for x in result))
if __name__ == "__main__": main()
