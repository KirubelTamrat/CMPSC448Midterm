# CMPSC 448 — Team review before submission

**Team listed in the report**
- Team leader: Kirubel Tamrat — **confirm this is the agreed team leader.**
- Team member: Chukwubikem Dara — **confirm spelling.**

## Assignment requirement check

| Requirement from `Midterm project-1-1.pdf` | Where it is addressed |
|---|---|
| Data with `(LLM_name, LLM_Input, LLM_output)` and at least three LLM families | `src/prepare_data.py`; report §2; ChatGPT, Claude, Gemini; 300 balanced examples |
| Explain dataset origin and construction; handle permissions | `README.md`, `data/DATA_SOURCE.md`, report §2 and §7 |
| Implement and train CNN and RNN | `src/models.py`, `src/train.py`; report §3 |
| RQ1: response-only source identification | Report §4; CNN and LSTM |
| RQ2: input-only, output-only, and input + output | Report §4, Figure 1 and six test rows |
| RQ3: unseen task type (extra credit) | Report §5; held-out writing category |
| RQ4: language / formatting signals (extra credit) | Report §5, Figure 3, style-only controls and format test |
| Problem definition, training details, results, analysis, lessons learned | Report §§1–7; figures and tables |
| Both team members' names in PDF | PDF title area |
| Code and report PDF stored in a GitHub repository | **Not completed by preparing this ZIP; you must upload the files** |
| Canvas submission is the GitHub URL | **You must submit the real repository link** |
| Optional presentation | No slides required if you choose the specified 17/20 non-presentation default |

## Checks performed for this package

- Original uploaded dataset workbook SHA-256: `9814d24eb2865c8bf03aad93d8ee2bab5333e55984bd07e30554b315586a3074`.
- Source workbook normalized to 100 unique shared prompts / 300 labeled answers; classes balanced.
- Training split checked: 69 prompt groups for train, 16 for validation, 15 for test, with no repeated prompt between groups.
- Saved CNN and LSTM response-only checkpoints reevaluated on the 45-answer holdout: **80.0%** and **55.6%** (same as the report).
- Style-only baseline reevaluated: 48.9% (word count) and 64.4% (seven features).
- Synthetic-software fixture ran successfully; its results are not used in the report.
- PDF regenerated from experiment results, with corrected chart scales and discussion of confusions, domain shift, limitations and model behavior.

## What the team still needs to verify

1. Both members should actually read the code/report and be able to explain CNN, LSTM, data split, and the six input/model combinations. The instructions explicitly make the team responsible for checking AI-assisted work.
2. Confirm the team leader, spelling, and add a brief description of genuine individual contributions **only if you can state them accurately** (not a required rubric item).
3. Confirm third-party dataset permissions before redistributing its complete text. The original workbook is intentionally omitted from the ZIP; `src/prepare_data.py` can retrieve it for local reproduction.
4. If desired, repeat the experiment across different random seeds or use a larger dataset. The current results are from one held-out split. These are worthwhile improvements, **not additional stated assignment requirements**.
5. Create/push a GitHub repository containing the source code, report PDF, README, and result plots. Original response records should not be committed without permission; the repo contains code to regenerate them locally.
6. Submit the repository URL to Canvas by the deadline. Decide whether to present; the PDF states a default 17/20 for not presenting.
