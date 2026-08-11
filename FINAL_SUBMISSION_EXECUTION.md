# Final submission execution handoff

This stage does **not** run new scientific experiments. The extended HMM and supervised ML results are already fixed and committed.

Work only on branch:

`review/regime-robustness-extension`

Do not change numerical results, assets, features, model-selection logic, conclusions, or literature claims unless a deterministic execution error proves that the source is invalid.

## 1. Update branch and environment

```bash
git fetch origin
git checkout review/regime-robustness-extension
git pull --ff-only origin review/regime-robustness-extension

source .venv/bin/activate 2>/dev/null || true
python --version
python -m pip install -r requirements.txt

git rev-parse HEAD
```

## 2. Static checks

```bash
python -m py_compile \
  build_final_notebook.py \
  patch_final_notebook.py \
  patch_protocol_clarification.py \
  run_canonical.py \
  run_extended_analysis.py \
  run_supervised_baselines.py \
  analyze_extended_context.py \
  src/*.py tests/*.py

python -m pytest tests/test_protocol_foundation.py tests/test_regime_extension.py -q
```

If a deterministic code/protocol test fails, stop and return the full traceback.

## 3. Rebuild the final notebook

Run the builder and both wording-only review patches, in this order:

```bash
python build_final_notebook.py
python patch_final_notebook.py
python patch_protocol_clarification.py
```

Confirm that they rewrite/update:

`HMM_Market_Regimes_Project.ipynb`

The final protocol clarification must make the following distinction explicit:

- HMM candidate selection: scaler/model fit on Train, selection on Validation;
- final selected HMM: refit on all pre-Test history (Train+Validation), then evaluated on Test with past-only inference;
- fixed Supervised ML baselines: fit on Train only; Validation is not used for tuning or refitting;
- therefore the supervised comparison is supplementary and does not use an identical amount of training history to the final HMM.

Then execute the notebook top-to-bottom **without rerunning the expensive experiments**:

```bash
jupyter nbconvert \
  --to notebook \
  --execute HMM_Market_Regimes_Project.ipynb \
  --output /tmp/HMM_Market_Regimes_Project_executed.ipynb \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=python3
```

Replace the repository notebook with the executed version:

```bash
cp /tmp/HMM_Market_Regimes_Project_executed.ipynb HMM_Market_Regimes_Project.ipynb
```

QA the executed notebook:

- no error outputs or tracebacks;
- all saved artifact paths resolve;
- all important tables/figures render;
- Hebrew Markdown remains readable RTL;
- no network download or HMM/supervised retraining occurs during notebook execution;
- the final conclusions are generated from the committed result artifacts;
- the fitting-history distinction above appears correctly in the notebook.

## 4. Compile the XeLaTeX report

From the repository root:

```bash
cd reports
xelatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
cd ..
```

The required output is:

`reports/report.pdf`

If compilation fails because of a LaTeX syntax/RTL/layout issue, fix only what is necessary for compilation and presentation. Do not change scientific numbers or conclusions.

After compilation inspect the log:

```bash
grep -Ei "undefined|overfull|underfull|warning|error" reports/report.log || true
pdfinfo reports/report.pdf | head -20
```

## 5. Visual PDF QA

If `pdftoppm` is available:

```bash
rm -rf /tmp/hmm_report_pages
mkdir -p /tmp/hmm_report_pages
pdftoppm -png -r 150 reports/report.pdf /tmp/hmm_report_pages/page >/dev/null 2>&1
ls -lh /tmp/hmm_report_pages | head
```

Inspect every rendered page if your environment supports visual inspection. At minimum verify:

- title page has Matan Eshel / 203502802 and no placeholders;
- Hebrew is RTL and English technical terms are not reversed;
- no clipped tables;
- no figures outside page boundaries;
- no broken formulas or black-square glyphs;
- captions are readable;
- bibliography renders;
- the Reinforcement Learning Future Work section is present and clearly labeled as future work, not an executed experiment;
- the protocol wording does not incorrectly claim that all final models use identical fitting history.

If you cannot visually inspect images, state that explicitly in the final handoff; do not claim visual QA was performed.

## 6. Final tests

With the data cache already present, run:

```bash
python -m pytest tests/ -q 2>&1 | tee /tmp/hmm_final_pytest.log
```

Classify any failures as:

1. deterministic code/protocol failure;
2. network/Yahoo dependency;
3. stale legacy-test assumption.

Do not hide failures.

## 7. Commit final generated artifacts

Review changes first:

```bash
git status --short
git diff --stat
```

Commit at least:

- `HMM_Market_Regimes_Project.ipynb`
- `reports/report.pdf`
- any minimal LaTeX/source fix required for successful compilation

Do not commit `.venv`, `data_cache`, `/tmp`, LaTeX auxiliary files, or rendered QA PNGs.

```bash
git add HMM_Market_Regimes_Project.ipynb reports/report.pdf
# Add source fixes only if you made any.
git add reports/report.tex reports/sections reports/references.bib \
  build_final_notebook.py patch_final_notebook.py patch_protocol_clarification.py \
  CHANGES.md FINAL_SUBMISSION_EXECUTION.md 2>/dev/null || true

git commit -m "Build final submission after protocol QA"
git push origin review/regime-robustness-extension
```

## 8. Return this execution report

Return:

- starting commit SHA;
- final commit SHA;
- Python version;
- static test result;
- notebook execution result;
- notebook cell count;
- report compilation result;
- PDF page count;
- whether visual QA was actually performed;
- full pytest pass/fail count and failure classification;
- TeX overfull/underfull counts;
- unresolved citation/reference count;
- any warnings or changes you had to make.

Do not merge branches. Stop after pushing the final artifacts so the review agent can perform one last content audit before merge to `main`.
