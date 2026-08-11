# Execution handoff — regime robustness branch

Use this file when a separate compute agent/server is asked to validate and run the project.

## Important constraints

- Work only on branch `review/regime-robustness-extension`.
- Do **not** redesign the experiment, change tickers, change dates, choose favorable seeds, or edit the report/notebook based on outcomes.
- Do **not** overwrite `experiments_canonical/canonical_20260810_014841/`.
- If Yahoo Finance fails or rate-limits, report it explicitly. Do not invent or substitute data silently.
- Preserve all stdout/stderr and exact commands used.
- A failed network-dependent test is not the same as a failed offline protocol test.

## 1. Get the branch and environment

```bash
git fetch origin
git checkout review/regime-robustness-extension
git pull --ff-only origin review/regime-robustness-extension

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If an existing clean environment already satisfies `requirements.txt`, it may be reused instead of recreating the venv.

Record:

```bash
python --version
python -m pip freeze > /tmp/hmm_pip_freeze.txt
git rev-parse HEAD
```

## 2. Syntax + offline protocol tests FIRST

Run these before any market download:

```bash
python -m py_compile run_canonical.py run_extended_analysis.py analyze_extended_context.py src/*.py tests/*.py
python -m pytest tests/test_protocol_foundation.py tests/test_regime_extension.py -q
```

If either command fails, **stop**. Return the complete traceback and do not start the expensive experiment.

## 3. Quick smoke test

Run:

```bash
python run_extended_analysis.py --quick 2>&1 | tee /tmp/hmm_extended_quick.log
```

Expected behavior:

- requests/caches the economically diverse asset panel;
- fits K=2/3/4 with seeds 42 and 123;
- skips walk-forward analysis;
- writes a new `experiments_extended/extended_*` directory;
- emits a complete `manifest.json` plus CSV/JSON/PNG diagnostics.

After it finishes, validate the latest run:

```bash
QUICK_RUN=$(ls -dt experiments_extended/extended_* | head -1)
echo "$QUICK_RUN"
python - <<'PY'
import json, pathlib, pandas as pd
run = sorted(pathlib.Path('experiments_extended').glob('extended_*'))[-1]
manifest = json.loads((run/'manifest.json').read_text())
print('status:', manifest['status'])
print('loaded_assets:', manifest.get('loaded_assets'))
print('data_load_failures:', manifest.get('data_load_failures'))
print('asset_status:', manifest.get('asset_status'))
print('artifacts:', len(manifest.get('artifacts', [])))
metrics = pd.read_csv(run/'test_metrics_all_assets.csv')
print(metrics.to_string(index=False))
PY
```

If the quick run has code/model failures (not merely a single Yahoo download failure), stop and return the logs/results.

## 4. Full extended run

Only after Steps 2–3 pass:

```bash
python run_extended_analysis.py 2>&1 | tee /tmp/hmm_extended_full.log
```

This is the main requested computation. It uses:

- SPY, QQQ, IWM, TLT, GLD, HYG, BTC-USD, JPM, NVDA;
- K=2,3,4;
- seeds 42,123,456,789,2026;
- posterior uncertainty diagnostics;
- seed stability via Adjusted Rand Index;
- hard-state and soft-posterior HMM predictions;
- state-conditioned error analysis;
- SPY-conditioned cross-asset analysis;
- three expanding-window robustness folds on SPY.

After completion:

```bash
FULL_RUN=$(ls -dt experiments_extended/extended_* | head -1)
echo "$FULL_RUN"
python analyze_extended_context.py --run-dir "$FULL_RUN" 2>&1 | tee /tmp/hmm_extended_context.log
```

The second command adds two post-hoc diagnostics that do **not** affect training/model selection:

1. external validation against `^VIX`;
2. empirical state dwell time vs the first-order HMM geometric-duration implication `1/(1-a_ii)`.

## 5. Full test suite AFTER data cache exists

Run:

```bash
python -m pytest tests/ -q 2>&1 | tee /tmp/hmm_pytest_full.log
```

If tests fail, classify failures into:

- deterministic code/protocol failures;
- Yahoo/network/rate-limit failures;
- stale legacy-test assumptions.

Do not simply report a pass/fail count without this distinction.

## 6. Return / commit the outputs

First provide a short run summary containing:

- exact Git commit SHA;
- Python version;
- quick-run directory;
- full-run directory;
- which assets loaded successfully;
- any Yahoo failures;
- offline test result;
- full pytest result;
- total runtime for the full extended run.

Then commit the **full extended run only**, not the quick smoke-test run and not `data_cache/`.

Because this repository historically ignores `*.csv` and `*.png`, force-add the selected full run:

```bash
git status --short
git add -f "$FULL_RUN"
git commit -m "Add verified extended HMM robustness results"
git push origin review/regime-robustness-extension
```

Do not commit `data_cache/`, `.venv/`, `/tmp` logs, or the quick-run directory.

If `analyze_extended_context.py` adds files after the first `git add`, make sure they are included in the same full-run commit.

## 7. Do NOT yet rebuild the report/notebook

Stop after pushing the verified full-run artifacts. The review agent will inspect the real numerical results first and only then update:

- `HMM_Market_Regimes_Project.ipynb`;
- `reports/report.tex` / PDF;
- final conclusions;
- submission package.

This prevents the narrative from being written before the evidence exists.
