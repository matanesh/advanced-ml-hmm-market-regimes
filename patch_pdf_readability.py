"""Final visual-readability patch for the Hebrew XeLaTeX report.

This patch changes presentation only. It does not modify experimental results,
methodology, numerical values, figures, or conclusions.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

PATCHES = {
    "reports/report.tex": [
        (r"\newcommand{\tech}[1]{\LR{#1}}", r"\DeclareRobustCommand{\tech}[1]{\LR{#1}}"),
        (r"\newcommand{\asset}[1]{\LR{#1}}", r"\DeclareRobustCommand{\asset}[1]{\LR{#1}}"),
        (
            "\\pagenumbering{roman}\n"
            "\\input{sections/00_abstract}\n"
            "\\clearpage\n"
            "\\tableofcontents\n"
            "\\thispagestyle{plain}\n"
            "\\clearpage\n"
            "\\pagenumbering{arabic}",
            "\\pagenumbering{roman}\n"
            "\\input{sections/00_abstract}\n"
            "\\clearpage\n"
            "\\begingroup\n"
            "\\small\n"
            "\\setlength{\\parskip}{0pt}\n"
            "\\tableofcontents\n"
            "\\endgroup\n"
            "\\thispagestyle{plain}\n"
            "\\clearpage\n"
            "\\pagenumbering{arabic}",
        ),
        (
            "\\clearpage\n"
            "\\bibliographystyle{plain}\n"
            "\\begingroup\n"
            "\\small\n"
            "\\bibliography{references}\n"
            "\\endgroup",
            "\\clearpage\n"
            "\\begingroup\n"
            "\\small\n"
            "\\begin{english}\n"
            "\\renewcommand{\\refname}{\\makebox[\\textwidth][r]{\\texthebrew{מקורות}}}\n"
            "\\bibliographystyle{plain}\n"
            "\\bibliography{references}\n"
            "\\end{english}\n"
            "\\endgroup",
        ),
    ],
    "reports/sections/00_abstract.tex": [
        (r"\textbf{latent market-state estimator}", r"\textbf{\tech{latent market-state estimator}}"),
    ],
    "reports/sections/02_data_protocol.tex": [
        (r"\section{נתונים, Features ופרוטוקול ניסוי}", r"\section{נתונים, \tech{Features} ופרוטוקול ניסוי}"),
        (r"\subsection{Feature Vector}", r"\subsection{\tech{Feature Vector}}"),
        (r"\subsection{שחזור ומעקב אחר provenance}", r"\subsection{שחזור ומעקב אחר \tech{provenance}}"),
    ],
    "reports/sections/03_method.tex": [
        (r"\subsection{Hidden Markov Model}", r"\subsection{\tech{Hidden Markov Model}}"),
        (r"\subsection{Hard state לעומת Soft posterior}", r"\subsection{\tech{Hard state} לעומת \tech{Soft posterior}}"),
        (r"\subsection{Persistence ומשך משטר}", r"\subsection{\tech{Persistence} ומשך משטר}"),
        ("את ה־Hidden State", r"את ה־\tech{Hidden State}"),
        ("ב־Gaussian HMM", r"ב־\tech{Gaussian HMM}"),
        ("דרך Gaussian emission", r"דרך \tech{Gaussian emission}"),
        ("באמצעות Baum-Welch / EM", r"באמצעות \tech{Baum-Welch / EM}"),
        ("Validation log-likelihood", r"\tech{Validation log-likelihood}"),
        ("Hidden Semi-Markov Model", r"\tech{Hidden Semi-Markov Model}"),
    ],
    "reports/sections/04_experiments.tex": [
        (r"\section{ניסויים, Baselines ומדדי הערכה}", r"\section{ניסויים, \tech{Baselines} ומדדי הערכה}"),
        (r"\subsection{Baselines פשוטים}", r"\subsection{\tech{Baselines} פשוטים}"),
        (r"\subsection{Supervised ML Baselines}", r"\subsection{\tech{Supervised ML Baselines}}"),
        (r"\subsection{ניסויי Robustness}", r"\subsection{ניסויי \tech{Robustness}}"),
        (r"\textbf{Naive - train mean:}", r"\textbf{\tech{Naive - train mean:}}"),
        (r"\textbf{Naive persistence:}", r"\textbf{\tech{Naive persistence:}}"),
        (r"\textbf{Moving Average 5:}", r"\textbf{\tech{Moving Average 5:}}"),
        (r"\textbf{Discrete Markov Chain:}", r"\textbf{\tech{Discrete Markov Chain:}}"),
    ],
    "reports/sections/05_regime_results.tex": [
        (r"\subsection{פירוש states ב־SPY}", r"\subsection{פירוש \tech{states} ב־\asset{SPY}}"),
        (r"\subsection{Posterior uncertainty סביב מעברים}", r"\subsection{\tech{Posterior uncertainty} סביב מעברים}"),
        (r"\subsection{External validation מול VIX}", r"\subsection{\tech{External validation} מול \asset{VIX}}"),
        (r"\subsection{Cross-asset behavior}", r"\subsection{\tech{Cross-asset behavior}}"),
        (r"\(K\) & Seed & Val. LL / obs. & BIC (Train) & Min occupancy (\%)", r"\(K\) & \tech{Seed} & \tech{Val. LL / obs.} & \tech{BIC (Train)} & \tech{Min occupancy (\%)}"),
        (r"State & פרשנות & שכיחות & תשואה & Vol. & Range & Mean DD & משך", r"\tech{State} & פרשנות & שכיחות & תשואה & \tech{Vol.} & \tech{Range} & \tech{Mean DD} & משך"),
        ("0 & Normal / moderate &", r"0 & \tech{Normal / moderate} &"),
        ("1 & Calm / low-vol &", r"1 & \tech{Calm / low-vol} &"),
        ("2 & Rare stress / crisis-like &", r"2 & \tech{Rare stress / crisis-like} &"),
        ("3 & Persistent high-vol &", r"3 & \tech{Persistent high-vol} &"),
        (r"Asset & Corr. State 0 & Corr. State 1 & Corr. State 3", r"\tech{Asset} & \tech{Corr. State 0} & \tech{Corr. State 1} & \tech{Corr. State 3}"),
    ],
    "reports/sections/06_prediction_results.tex": [
        (r"\subsection{שגיאת magnitude: דוגמת SPY}", r"\subsection{שגיאת \tech{magnitude}: דוגמת \asset{SPY}}"),
        (r"\subsection{Hard state לעומת Soft posterior}", r"\subsection{\tech{Hard state} לעומת \tech{Soft posterior}}"),
        (r"\subsection{Walk-forward robustness על SPY}", r"\subsection{\tech{Walk-forward robustness} על \asset{SPY}}"),
        (r"Model & Mean DPA (\%)", r"\tech{Model} & \tech{Mean DPA (\%)}"),
        ("Naive - train mean & 54.54", r"\tech{Naive - train mean} & 54.54"),
        ("Logistic Regression & 54.14", r"\tech{Logistic Regression} & 54.14"),
        ("Discrete Markov Chain & 53.90", r"\tech{Discrete Markov Chain} & 53.90"),
        ("Random Forest Classifier & 53.27", r"\tech{Random Forest Classifier} & 53.27"),
        ("Gaussian HMM - soft posterior & 53.06", r"\tech{Gaussian HMM - soft posterior} & 53.06"),
        ("HistGradientBoostingClassifier & 52.49", r"\tech{HistGradientBoostingClassifier} & 52.49"),
        ("Gaussian HMM - hard state & 52.06", r"\tech{Gaussian HMM - hard state} & 52.06"),
        ("Naive - persistence & 50.45", r"\tech{Naive - persistence} & 50.45"),
        ("Moving Average 5 & 50.27", r"\tech{Moving Average 5} & 50.27"),
        (r"Model & DPA (\%) & MAE return & RMSE return & MAPE price (\%)", r"\tech{Model} & \tech{DPA (\%)} & \tech{MAE return} & \tech{RMSE return} & \tech{MAPE price (\%)}"),
        ("Naive - train mean & 58.50", r"\tech{Naive - train mean} & 58.50"),
        ("Naive - persistence & 53.16", r"\tech{Naive - persistence} & 53.16"),
        ("Moving Average 5 & 53.16", r"\tech{Moving Average 5} & 53.16"),
        ("Discrete Markov Chain & 58.50", r"\tech{Discrete Markov Chain} & 58.50"),
        ("Gaussian HMM - hard state & 58.50", r"\tech{Gaussian HMM - hard state} & 58.50"),
        ("Gaussian HMM - soft posterior & 58.50", r"\tech{Gaussian HMM - soft posterior} & 58.50"),
    ],
    "reports/sections/07_discussion.tex": [
        (r"\subsection{Expressiveness מול robustness}", r"\subsection{\tech{Expressiveness} מול \tech{robustness}}"),
        (r"\subsection{משך המשטר והנחת Markov}", r"\subsection{משך המשטר והנחת \tech{Markov}}"),
        (r"\textbf{conditional market context}", r"\textbf{\tech{conditional market context}}"),
        (r"\textbf{latent state estimation}", r"\textbf{\tech{latent state estimation}}"),
        (r"\textbf{point forecasting}", r"\textbf{\tech{point forecasting}}"),
    ],
    "reports/sections/08_rl_future_work.tex": [
        (r"\section{הרחבה עתידית: Regime-aware Reinforcement Learning}", r"\section{הרחבה עתידית: \tech{Regime-aware Reinforcement Learning}}"),
        (r"\subsection{ניסוח כ־Markov Decision Process}", r"\subsection{ניסוח כ־\tech{Markov Decision Process}}"),
    ],
    "reports/sections/09_limitations_conclusion.tex": [
        (r"\textbf{modeling conditional market structure}", r"\textbf{\tech{modeling conditional market structure}}"),
        (r"\textbf{point forecasting}", r"\textbf{\tech{point forecasting}}"),
        ("Regime-aware Reinforcement Learning", r"\tech{Regime-aware Reinforcement Learning}"),
    ],
}


def main() -> None:
    total = 0
    for rel_path, replacements in PATCHES.items():
        path = ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        for old, new in replacements:
            count = text.count(old)
            if count != 1:
                raise RuntimeError(
                    f"Expected exactly one match in {rel_path}: {old!r}; got {count}"
                )
            text = text.replace(old, new)
            total += 1
        path.write_text(text, encoding="utf-8")
        print(f"Patched {rel_path}: {len(replacements)} replacement(s)")
    print(f"Applied {total} visual-readability replacement(s).")


if __name__ == "__main__":
    main()
