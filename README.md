# LHC resume-state parser

> A 220-line Python parser (≈80 lines of load-bearing logic) that scores 2/2 deterministically on two LHC v0.2 resumption tasks where the fine-tuned Ember v0.1.5 model averaged 1.0–1.25 across 12 cells. It uses no LLM in the forward pass.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Benchmark](https://img.shields.io/badge/🤗%20Benchmark-hugonogueira%2Flhc--v0.2-yellow)](https://huggingface.co/datasets/hugonogueira/lhc-v0.2)

## What this is

A deterministic baseline for the [LHC v0.2 resumption suite](https://github.com/hugomn/lhc) — picking up a half-finished workflow after a long context gap. It detects structured "next step" markers (`NEXT:`, `[NEXT]`, "next planned action") and inconsistencies between recorded decisions and next-action notes (e.g. `RECIPE-A` vs `RECIPE-B` referenced in different sections of the same loaded state).

## What it is NOT

- Not a model. It's regex.
- Not a chatbot. The forward pass is `re.findall`.
- Not a general benchmark solver. It targets *resumption* tasks where state is structured (workflow trackers, payload logs). It explicitly declines on free-prose continuations.

## Why it exists

The [LHC v0.2 methodology arc](https://github.com/hugomn/lhc/tree/main/docs/journal) asks whether fine-tuning is even the right layer for resumption-style failures. This parser is the empirical answer for our specific benchmark: **on two structured-state resumption tasks, a 220-line deterministic script scores 2/2 reliably while every fine-tuned 8B model we tested fluctuates between 0/2 and 2/2 across trials.** The comparison:

`resumption_v2_004` (warehouse pick-pack tracker — a structured workflow with `[NEXT]` tags):

| Solver | Per-trial scores across 12 cells (4 gap modes × 3 trials) | Mean |
|---|---|---:|
| **Parser (this repo)** | `2/2` (deterministic, single run, no trial variance applies) | **2.00** |
| Llama-3.1-8B | `1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2` | 1.92 |
| Ministral-8B-2512 | `2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2` | 1.67 |
| Qwen3-8B base | `2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2` | 1.50 |
| Ember v0.1.5 (fine-tune) | `2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1` | 1.25 |

`resumption_v2_008` (bakery inconsistency detection — `RECIPE-A` decision vs `RECIPE-B` next-action mismatch):

| Solver | Per-trial scores across 12 cells | Mean |
|---|---|---:|
| **Parser (this repo)** | `2/2` (deterministic) | **2.00** |
| Ministral-8B-2512 | `2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2` | 1.92 |
| Qwen3-8B base | `2, 0, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2` | 1.50 |
| Llama-3.1-8B | `0, 0, 0, 2, 2, 2, 1, 2, 1, 2, 0, 2` | 1.17 |
| Ember v0.1.5 (fine-tune) | `0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2` | 1.00 |

Two observations:

1. **The parser ties or beats every model on the mean for these tasks.** Ministral and Llama do match the parser's max score of 2/2, but with cross-trial variance the parser doesn't have. For resumption tasks where the state is structured, you can get the same score from regex that you get from an 8B-class LLM — at zero inference cost and zero variance.
2. **Ember v0.1.5 has the lowest mean on both tasks.** The fine-tune that was supposed to improve on Qwen3-8B's resumption capability instead drops below it on the structured-state tasks. The parser isn't the comparison that matters for the methodology arc — the *base model* is — but the parser sets a useful floor.

Implication: for production agent workflows where the inter-session state is structured, fine-tuning may be the wrong layer. A schema validator + parser does the job better, deterministically, and at zero inference cost.

## Full LHC v0.2 resumption scorecard

8 tasks, judged by Claude Opus 4.7 against the same rubric the LLM entrants use. Score = `correctness + meta_awareness` ∈ {0, 1, 2}.

| Task | Score | What happened |
|---|---:|---|
| `resumption_v2_001` (lab notebook reload) | 1/2 | Detected the lab `[NEXT]` marker; scored correct on meta_awareness, missed the substantive correctness bar |
| `resumption_v2_002` (NICU feeding titration) | 0/2 | No structured marker; declined as designed (the parser does not free-associate) |
| `resumption_v2_003` (jewelry chat-log) | 0/2 | Free-prose; declined |
| `resumption_v2_004` (warehouse tracker) | **2/2** | Detected `[NEXT]` tag |
| `resumption_v2_005` (loom weight) | 1/2 | Detected "next planned action" label |
| `resumption_v2_006` (ATC handoff log) | 0/2 | No marker the parser recognizes; declined |
| `resumption_v2_007` (multi-thread editor) | 0/2 | Free-prose; declined |
| `resumption_v2_008` (bakery batch inconsistency) | **2/2** | Detected `RECIPE-A`/`RECIPE-B` mismatch |

**Overall: 6/16 = 0.75 / 2.00 mean.** Full scorecard with judge reasoning at [`scorecard.json`](scorecard.json).

For reference, on the same 8 resumption tasks under the original LHC v0.2 sweep (4 gap modes × 3 trials = 12 cells per task), the fine-tuned Ember v0.1.5 averaged **1.21 / 2.00** mean — higher overall than the parser, but with mean **1.25 on `resumption_v2_004`** and mean **1.00 on `resumption_v2_008`** (vs the parser's 2.0 on both). Across all four tested 8B models (Ember, Qwen3-8B base, Ministral-8B-2512, Llama-3.1-8B), every model had a mean below 2.0 on at least one of those two structured-state tasks. See the [LHC v0.2 dataset](https://huggingface.co/datasets/hugonogueira/lhc-v0.2) for all numbers.

## Install

```bash
git clone https://github.com/hugomn/lhc-resume-state-parser.git
cd lhc-resume-state-parser
pip install pyyaml          # only runtime dep
pip install pytest          # only if you want to run the tests
```

## Use

```bash
# CLI: feed it any LHC v0.2 task YAML
python cli.py examples/resumption_v2_004.yaml

# Same, but also print the task's setup + probe before the response
python cli.py --show-task examples/resumption_v2_004.yaml
```

```python
# Library
import yaml
from parser import parse_task

task = yaml.safe_load(open("examples/resumption_v2_004.yaml"))
print(parse_task(task))
# → "Per the loaded state, the next step is: L5, SKU C-3308, mid-weight base layers, qty 8…"
```

## Tests

```bash
pytest tests/
```

9 tests covering the NEXT-step pattern library (4 cases), the inconsistency detector (3 cases), and end-to-end on the bundled example task plus the decline path.

## How to think about this result

The parser **scores 2/2 deterministically on 2 of 8 resumption tasks** and **declines or scores low on the other 6**. It is not a benchmark-beater; it is a **floor**.

The headline isn't that the parser beats every model — Ministral and Llama match its max score of 2/2 on these tasks. The headline is two things:

1. **Determinism.** The parser produces 2/2 on every run; the LLMs produce 2/2 on *some* trials and lower on others. If you need a reliable resumption signal in production, regex on structured state gives it to you.
2. **A floor on what "fine-tuning is worth it" means.** If your fine-tune's mean score on `resumption_v2_004` or `resumption_v2_008` is below 2.0 — and across the four 8B-class models we tested, *all* of them have means below 2.0 on at least one of these tasks — then a 220-line script is matching or beating it on a task where you're paying inference cost.

Where the parser fails: free-prose continuations (`_002` NICU titration, `_003` jewelry chat-log, `_006` ATC handoff, `_007` multi-thread editor). Those are where LLM capability shows up and a regex can't.

The methodology takeaway: the parser sets a floor. Any fine-tune that doesn't clear it on the structured-state tasks is not justifying its cost over the script.

## Reproduce on the full LHC v0.2 benchmark

The parser is also bundled inside the main LHC v0.2 repo at [`evals/v0.2/deterministic_baseline.py`](https://github.com/hugomn/lhc/blob/main/evals/v0.2/deterministic_baseline.py), where it runs end-to-end with the same harness the LLM entrants use. To regenerate `scorecard.json` from scratch:

```bash
git clone https://github.com/hugomn/lhc.git
cd lhc
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env  # ANTHROPIC_API_KEY for the Opus judge
python evals/v0.2/deterministic_baseline.py \
    --judge-provider anthropic \
    --output evals/results/v0.2/deterministic-baseline.json
```

Cost: ~$0.10 in Opus judge calls. No model API for the target — that's the whole point.

## License

Apache-2.0. Fork it. Run it on your own benchmark. If your benchmark has structured-state resumption tasks and a 220-line script beats your model, that's worth knowing.

## Citation

```bibtex
@misc{nogueira2026lhc-parser,
  title  = {LHC resume-state parser — a deterministic baseline for LHC v0.2 resumption tasks},
  author = {Nogueira, Hugo},
  year   = {2026},
  month  = {05},
  url    = {https://github.com/hugomn/lhc-resume-state-parser},
  note   = {Companion to LHC v0.2: https://huggingface.co/datasets/hugonogueira/lhc-v0.2}
}
```
