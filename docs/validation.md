# Release validation

The released metric implementation and parser retain the original source. Local verification covered:

- Frozen evaluator regression suite: 54 tests passed.
- Relocated window generator: 52 tests passed.
- Shared API orchestration: 27 tests and two subtests passed, using mocked requests.
- Fifteen complete example prompt/answer pairs across five settings reconstructed exactly using the production formatting functions.
- Missing, invalid, empty and repeated prediction histories checked through the standalone evaluator.
- Five full metadata settings verified for record counts, shared IDs and consistency with the original benchmark inputs.
- Five existing gateway one-minute baselines reproduced historical task score fields exactly without new inference.
- GT export and the original report CLI ran on the small examples.
- Repository payload checked for local absolute paths, credentials, caches and video/model files.

Run the scoring and data-format checks from the repository root:

```bash
python scripts/verify_examples.py
python scripts/check_release.py
cd evaluation/eval && pytest -q
```

The validation does not replace the historical human annotation process. Paid APIs and full local model inference were not rerun. Exact external model/environment revisions and the missing Google runner are documented in `baselines/README.md`. Annotation staffing and historical adjudication details remain to be confirmed by the authors; no additional review coverage is inferred.

The metadata-only downloader was exercised against `zacharyz21/EventSTVG` revision `599ee6c18b94df0daddd9d178a5c0d1eaf8dffcf`, using an approved account. Oracle metadata contained 3,516 unique records. The three repository example IDs were selected from the downloaded metadata and scored successfully with the independent evaluator. No video was downloaded during this check.
