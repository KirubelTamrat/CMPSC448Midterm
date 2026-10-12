# Input dataset (not redistributed)

Original source: https://github.com/Pranavsakunde/LLM-Response-Evaluation-Benchmark

The project results were computed from the user-supplied `dataset.xlsx` workbook.
SHA-256: `9814d24eb2865c8bf03aad93d8ee2bab5333e55984bd07e30554b315586a3074`

The workbook contains 100 shared prompts and 300 populated model responses (100 per class). Dataset creator claims these are actual ChatGPT, Claude, and Gemini outputs. API generation logs and exact provider model versions are not available here.

To reproduce: download the public `dataset.xlsx`, save as `data/source_dataset.xlsx`, then run:

```bash
python -m src.prepare_data
python -m src.train
python -m src.style_baseline
python -m src.make_report
```

Do not add original dataset to a public GitHub repository without checking reuse rights. Its responses are omitted from the prepared archive.
