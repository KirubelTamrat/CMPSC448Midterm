# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 | Midterm project | Individual project, Kirubel Tamrat

## What this repository does

The goal is to predict whether a response was written by **ChatGPT, Claude, or Gemini**. Both of the required architectures are implemented directly in PyTorch: a multi-kernel 1D **CNN** and a bidirectional **LSTM**. The pipeline addresses all four research questions:

- **RQ1:** Identify the model from its response using the CNN and LSTM.
- **RQ2:** Compare prompt-only, response-only and prompt + response for each network.
- **RQ3:** Extra credit: train on non-writing categories and test on unseen writing questions.
- **RQ4:** Extra credit: quantify vocabulary, length and formatting features; strip Markdown at test time to check sensitivity.

The model comparisons use **prompt-group disjoint** train, validation and test partitions. Every model family is represented on every retained prompt; this avoids a common confound in which the prompt itself accidentally reveals which model was used. Test accuracy and macro-F1 are reported; chance accuracy with three balanced classes is 33.3%.

## Data provenance and permissions

The completed experiment uses the public [LLM Response Evaluation Benchmark](https://github.com/Pranavsakunde/LLM-Response-Evaluation-Benchmark) by Pranavsakunde. Its README describes **100 identical prompts** answered by **ChatGPT, Claude, and Gemini**, covering seven domains: general knowledge, coding, math, writing, reasoning, translation, and safety.

**IMPORTANT:** The original publisher, not this repo, collected these responses. Verify the data's terms and source accuracy. No third-party response corpus is redistributed here. The preparation script downloads it for local analysis or accepts a copy you download yourself. The code does **not** generate or pass off fabricated model responses as authentic. The synthetic fixture in `tests/` is *only* for checking software behavior, and fixture results are not research results.

The supplied original source workbook is not packaged. The source's public README does not state all model version and decoding details; report such unknowns as limitations.

## Actual held-out results (seed 42)

The completed training run used 69 train prompts, 16 validation prompts, and 15 test prompts, with three model responses per prompt. Each family therefore contributes 15 of the 45 test responses. Labels cannot be inferred from a prompt by construction, since every prompt was answered by all three models.

| Architecture | Prompt only | Response only | Prompt + response |
|---|---:|---:|---:|
| CNN | 33.3% | **80.0%** | 75.6% |
| Bidirectional LSTM | 33.3% | 55.6% | 51.1% |

CNN response-only macro-F1: **0.797**; LSTM response-only macro-F1: **0.557**. A chance-level baseline is **33.3%**. Results are based on a single small hold-out split, not a cross-validated benchmark. RQ3 cross-domain performance with writing held out: CNN **42.2%**, LSTM **53.3%**. RQ4 controls: word-count-only logistic regression **48.9%**, seven interpretable style features **64.4%**. Removing Markdown/list marks at test time changed CNN response-only accuracy from 80.0% to 77.8%, and LSTM from 55.6% to 57.8%.

Test predictions were measured on the user-provided `dataset.xlsx` and included in `results/experiment_results.json`. The dataset's original authors claim these are authentic model responses, but the exact model versions and generation configurations were not independently verifiable. Do not treat these results as proof of real-world authorship. The full numerical methodology is in the PDF.

## Quick start (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# One-time: download benchmark into data/source_dataset.xlsx and normalize it
python -m src.prepare_data --download

# Run all six required trials + both extra-credit investigations
python -m src.train

# Additional interpretable response-style baseline
python -m src.style_baseline

# Generate results-driven PDF
python -m src.make_report
```

If your Python environment cannot retrieve GitHub raw files, download the original `dataset.xlsx` from its [repository](https://github.com/Pranavsakunde/LLM-Response-Evaluation-Benchmark) and save it as `data/source_dataset.xlsx`. Then run `python -m src.prepare_data` without `--download`.

If the source workbook has unusual header names, `prepare_data.py` prints its column names. You can override auto-detection, for example:

```bash
python -m src.prepare_data --prompt-column "Prompt" --category-column "Category" \
    --gpt-column "ChatGPT Response" --claude-column "Claude Response" --gemini-column "Gemini Response"
```

Column labels in this example are **illustrative**, not verified against the source workbook.

## Files

| Path | Contents |
|---|---|
| `src/prepare_data.py` | Public workbook downloader, normalization, labeling |
| `src/models.py` | CNN and bidirectional LSTM layers |
| `src/train.py` | Prompt-group splitting, training, six required experiments, RQ3, RQ4, charts |
| `src/make_report.py` | Full report PDF generated from the actual saved training results |
| `src/style_baseline.py` | Additional logistic-regression style controls for RQ4 |
| `src/predict.py` | Optional demo: classify any text using a saved neural checkpoint |
| `tests/fixture_demo.csv` | **Synthetic, not valid for evaluation**; code test only |
| `report/Midterm_Project_Report.pdf` | Results-based report with all four research questions |
| `results/` | Populated after real training: JSON, CSVs, PNG charts, model checkpoints |

For a quick *software-only* smoke test, without pretending to have real model responses:

```bash
python -m src.train --data tests/fixture_demo.csv --out test_results --epochs 1 --skip-extras
```

## Technical details and experimental design

The input tokenizer lowercases and splits words, numbers and punctuation. The vocabulary is constructed from the training partition **only**. Responses are truncated or padded to 256 tokens. The CNN uses 3/5/7-width convolutional filters and max pooling. The LSTM is bidirectional and uses final hidden states. Both models have trainable 64-dimensional embeddings, dropout 0.3, cross entropy, Adam (learning rate 0.002), batch size 32, gradient clipping, maximum 14 epochs, and validation-loss-based early stopping. Seed: 42.

**Data split:** approximately 70/15/15 percent by *unique prompt*, not by response row, using GroupShuffleSplit. A separate leave-writing-category-out evaluation is used in RQ3. RQ4 measures mean response length, lexical diversity, average word size, Markdown marks, bullets, and newlines; the ablation removes visible markup **only at test time** to examine sensitivity of the already-trained networks. This is not a causal explanation of their internal features.

## Honest limitations

- **Small sample size:** 100 prompts / 300 responses before filtering. Neural models may have high variance.
- **Limited label set:** Three families only; unknown-family outputs will still be assigned to one of them.
- **Truncated text:** Models only see the first 256 tokens, not every word of lengthy responses.
- **Domain differences:** Distinguishing models on this curated benchmark is not proof of universal model authorship.
- **Version uncertainty:** Original source model version, temperature and generation metadata may not be available.
- **Source permissions:** Do not commit the original workbook or text records without verifying appropriate rights.
- **Individual ownership:** The course PDF allows AI to assist but says each student must understand and verify what they submit. Re-run the work and review interpretations yourself.

## Course submission

1. Run the project on **real** responses and inspect the output JSON/figures for errors.
2. Read, check and personalize `report/Midterm_Project_Report.pdf` after it is generated. Do not submit the separately marked DRAFT.
3. Push the source code, report PDF and figures to a GitHub repository. Do **not** push original response text or the original workbook without confirming reuse rights.
4. Submit that repository URL on Canvas. Presentation is optional in the assignment PDF; declining means the default 17/20 presentation credit per the document.

This project was trained and evaluated using the workbook supplied in this conversation. The repo ZIP has not been pushed to GitHub. The original response workbook and normalized CSV are not redistributed.
