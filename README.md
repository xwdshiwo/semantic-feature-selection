# Semantic Knowledge-Guided Sparse Feature Selection

The method combines evolutionary feature selection with proposal–critique agents, adaptive knowledge weights, and temporary population expansion.

## Setup

Install the packages listed in `requirements.txt`. For semantic proposals, configure a Chat Completions-compatible endpoint supporting JSON output and `enable_thinking`:

```sh
export LLM_API_KEY="your-api-key"
export LLM_API_URL="your-chat-completions-endpoint"
export LLM_MODEL="qwen-plus"
```

## Input

Provide an NPZ file containing:

- `X`: numeric matrix, samples × features.
- `y`: binary labels (0 and 1).
- `sample_ids`: unique sample identifiers.
- `feature_ids`: unique gene or miRNA identifiers matching the columns of `X`.
- `groups` (optional): donor identifiers for grouped cross-validation.
- `outer_fold` (optional): predefined fold assignments, numbered 0–4.

Use numeric or string arrays, not object arrays. Supply expression values on the intended analysis scale; imputation and standardization are fitted within each training fold.

Provide module memberships as a JSON list with zero-based feature indices:

```json
[{"feature_index": 0, "module": "module_name"}]
```

Set the disease context and experiment parameters in `config.json`. Without module memberships, the method uses data-driven population search.

## Run

```sh
python main.py --data dataset.npz --knowledge knowledge.json --output results
```

The output directory must not already exist. Defaults are five outer folds, three inner folds, logistic regression, and 200 feature-subset evaluations per outer fold. Agent prompts are in `agents.py`.

## Output

Each fold contains metrics, sample predictions, selected features, search curves, split indices, and agent records. `metrics.csv` summarizes the outer-fold results.

In `evolution.csv`, `best_score` is the best score in the current population archive; `raw_best_score` is the best score among all evaluated candidates. Both are internal search scores, separate from outer-fold test metrics. Check `agent_events.json` for skipped or failed API calls.
