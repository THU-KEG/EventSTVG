The official evaluator and frozen parser are retained in `eval/` and `bench/` with their original relative import structure. Run the standalone file interface from the release root:

```bash
python evaluation/evaluate.py --setting context_1min \
  --metadata examples/context_1min.metadata.jsonl \
  --predictions examples/predictions/copy_gt/context_1min.jsonl \
  --out artifacts/quickstart.json
```

See [the complete evaluation contract](../docs/evaluation.md), and [the frozen policy](eval/EVAL_POLICY.md).
