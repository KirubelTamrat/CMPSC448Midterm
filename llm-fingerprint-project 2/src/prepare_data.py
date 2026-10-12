"""Download and normalize a published 100-prompt, 3-LLM comparison workbook.

The workbook is not bundled because reuse permissions must be checked with the
original publisher; the project is not its publisher or author.
"""
from __future__ import annotations
import argparse
import re
import urllib.request
from pathlib import Path
import pandas as pd

SOURCE = "https://github.com/Pranavsakunde/LLM-Response-Evaluation-Benchmark"
RAW_URL = "https://raw.githubusercontent.com/Pranavsakunde/LLM-Response-Evaluation-Benchmark/main/dataset.xlsx"


def standardize_col(v):
    return re.sub(r"[^a-z0-9]+", "", str(v).lower())


def detect(cols, aliases, exclude=()):
    x = [(col, standardize_col(col)) for col in cols]
    for alias in aliases:
        m = [c for c, z in x if alias in z and not any(b in z for b in exclude)]
        if len(m) == 1:
            return m[0]
    return None


def load_excel(path: Path, **overrides) -> pd.DataFrame:
    wb = pd.ExcelFile(path)
    sheet = next((s for s in wb.sheet_names if s.strip().lower() == "dataset"), wb.sheet_names[0])
    df = pd.read_excel(path, sheet_name=sheet).dropna(how="all")
    print(f"Sheet: {sheet}. Columns: {list(df.columns)}")
    cols = list(df.columns)
    prompt = overrides.get("prompt_column") or detect(cols, ["prompt", "question", "userinput", "input"])
    category = overrides.get("category_column") or detect(cols, ["category", "tasktype", "domain"])
    model_cols = {
        "ChatGPT": overrides.get("gpt_column") or detect(cols, ["chatgptresponse", "chatgptanswer", "chatgptoutput", "chatgpt", "gptresponse", "gptanswer"]),
        "Claude": overrides.get("claude_column") or detect(cols, ["clauderesponse", "claudeanswer", "claudeoutput", "claude"]),
        "Gemini": overrides.get("gemini_column") or detect(cols, ["geminiresponse", "geminiansw", "geminioutput", "gemini"]),
    }
    # Prevent accidental use of the *scores* as the response texts.
    for model, col in model_cols.items():
        if col is not None and any(s in standardize_col(col) for s in ("score", "rating", "total", "winner", "clarity", "helpfulness")):
            model_cols[model] = None
    if not prompt or any(x is None for x in model_cols.values()):
        raise ValueError(
            "Could not safely detect prompt and all three response columns. "
            f"Found: prompt={prompt}, category={category}, responses={model_cols}. "
            "Run with --prompt-column, --gpt-column, --claude-column, and --gemini-column "
            "using names from the printed column list. Do not substitute evaluation scores for responses."
        )
    rows = []
    for i, row in df.iterrows():
        question = row[prompt]
        if pd.isna(question) or len(str(question).strip()) < 4:
            continue
        cat = str(row[category]).strip() if category and pd.notna(row[category]) else "unclassified"
        for model, col in model_cols.items():
            val = row[col]
            if pd.isna(val) or len(str(val).strip()) < 15:
                continue
            rows.append({"prompt_id": f"p_{int(i):04d}", "LLM_name": model,
                         "LLM_Input": str(question).strip(), "LLM_output": str(val).strip(),
                         "category": cat})
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No real text responses found. Inspect the workbook and column mapping.")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workbook", type=Path, default=Path("data/source_dataset.xlsx"))
    p.add_argument("--out", type=Path, default=Path("data/llm_responses.csv"))
    p.add_argument("--download", action="store_true", help="Fetch the public source workbook from its publisher")
    for name in ["prompt", "category", "gpt", "claude", "gemini"]:
        p.add_argument(f"--{name}-column", default=None)
    args = p.parse_args()
    args.workbook.parent.mkdir(exist_ok=True, parents=True)
    if args.download and not args.workbook.exists():
        print("Downloading original research workbook from:", RAW_URL)
        try:
            urllib.request.urlretrieve(RAW_URL, args.workbook)
        except Exception as e:
            raise SystemExit(f"Download failed ({e}). Download dataset.xlsx from {SOURCE} manually and save it at {args.workbook}")
    if not args.workbook.exists():
        raise SystemExit(f"Workbook is missing at {args.workbook}. Run --download or put the XLSX there yourself.")
    overrides = {f"{key}_column": getattr(args, f"{key}_column") for key in ("prompt", "category", "gpt", "claude", "gemini")}
    out = load_excel(args.workbook, **overrides)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} labeled records and {out.prompt_id.nunique()} prompts to {args.out}")
    print(out.groupby(["LLM_name", "category"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
