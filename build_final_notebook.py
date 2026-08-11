from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "HMM_Market_Regimes_Project.ipynb"


def md(text: str):
    text = textwrap.dedent(text).strip()
    return nbf.v4.new_markdown_cell(f'<div dir="rtl">\n\n{text}\n\n</div>')


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = []

cells.append(md(r'''
# זיהוי משטרי שוק באמצעות Gaussian HMM

## פרויקט גמר בקורס "שיטות מתקדמות בלמידת מכונה"

**מתן אשל, 203502802**

המחקר בוחן שתי שאלות נפרדות אך קשורות: האם `Gaussian HMM` מסייע בחיזוי כיוון התשואה ביום הבא, והאם הוא מצליח לגלות **מבנה לטנטי יציב ומשמעותי של משטרי שוק**. ההבחנה חשובה: מודל יכול להיות שימושי לתיאור מצב השוק גם אם אינו מנבא באופן עקבי את היום הבא.

הגרסה הסופית כוללת תשעה נכסים מייצגים: `SPY`, `QQQ`, `IWM`, `TLT`, `GLD`, `HYG`, `BTC-USD`, `JPM`, `NVDA`. בנוסף ל-HMM עם \(K\in\{2,3,4\}\), נבדקים Baselines פשוטים ומודלי Supervised ML (`Logistic Regression`, `Random Forest`, `HistGradientBoosting`). המחברת קוראת את תוצרי הריצות המאומתות ואינה מקשיחה מאות מספרים בתוך התאים.
'''))

cells.append(md(r'''
## 1. תצורת המחקר ומקורות התוצאות

שתי ריצות משמשות בסיס לניתוח הסופי:

- `extended_20260811_121957` — HMM מורחב, posterior uncertainty, seed stability, walk-forward, VIX ו-cross-asset analysis.
- `supervised_20260811_133344` — Logistic Regression, Random Forest ו-HistGradientBoosting על אותם Features ואותו split.

הנתונים התבקשו לטווח `2014-01-01` עד `2024-12-31`. החלוקה כרונולוגית: כ-70% Train, כ-15% Validation וכ-15% Test. שורות גבול שבהן יעד היום הבא חוצה partition הוסרו, וכל scaling מותאם על Train בלבד.
'''))

cells.append(code(r'''
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

PROJECT_ROOT = Path.cwd().resolve()
EXTENDED_RUN = PROJECT_ROOT / "experiments_extended" / "extended_20260811_121957"
SUPERVISED_RUN = PROJECT_ROOT / "experiments_supervised" / "supervised_20260811_133344"

assert (PROJECT_ROOT / "src").is_dir(), "יש להריץ את המחברת משורש הפרויקט."
assert EXTENDED_RUN.is_dir(), f"חסרה תיקיית HMM: {EXTENDED_RUN}"
assert SUPERVISED_RUN.is_dir(), f"חסרה תיקיית supervised: {SUPERVISED_RUN}"

with (EXTENDED_RUN / "manifest.json").open(encoding="utf-8") as f:
    hmm_manifest = json.load(f)
with (SUPERVISED_RUN / "manifest.json").open(encoding="utf-8") as f:
    supervised_manifest = json.load(f)

assert hmm_manifest["status"] == "completed"
assert supervised_manifest["status"] == "completed"
assert hmm_manifest["data_load_failures"] == []
assert supervised_manifest["data_load_failures"] == []

ASSETS = hmm_manifest["config"]["assets"]
FEATURES = hmm_manifest["config"]["feature_cols"]

print("HMM run:", hmm_manifest["run_id"])
print("Supervised run:", supervised_manifest["run_id"])
print("Assets:", ", ".join(ASSETS))
print("Features:", ", ".join(FEATURES))
print("Python:", hmm_manifest["package_versions"]["python"])
'''))

cells.append(md(r'''
הבדיקות בתא אינן קוסמטיות: הן מוודאות שהמחברת נשענת על ריצות שסומנו `completed`, שכל הנכסים נטענו ושאין כשלי דאטה שקטים. הפאנל נבחר כדי לייצג התנהגויות כלכליות שונות ולא כדי לבחור בדיעבד מניות שמציגות תוצאה יפה.

`SPY` מייצג שוק מניות רחב, `QQQ` צמיחה/טכנולוגיה, `IWM` small caps, `TLT` אג"ח ארוכות, `GLD` זהב, `HYG` אשראי high-yield, `BTC-USD` נכס אלטרנטיבי תנודתי, ו-`JPM` ו-`NVDA` מייצגות פיננסים ומניית growth/high-beta.
'''))

cells.append(md(r'''
## 2. Feature Vector ומניעת leakage

לכל יום נבנה וקטור:

\[
x_t=[r_t,\sigma_t^{(20)},\mathrm{Range}_t,\Delta\log V_t]
\]

כאשר \(r_t=\log(P_t/P_{t-1})\), התנודתיות מחושבת על 20 ימי עבר, ה-daily range מתאר את הטווח התוך-יומי, ו-volume change מתאר שינוי לוגריתמי בנפח. אלה ארבעה היבטים פשוטים של מצב השוק: כיוון, volatility, range ופעילות.
'''))

cells.append(code(r'''
qa = pd.DataFrame(supervised_manifest["qa"])
assert qa["target_safe_split"].all()
assert (~qa["shuffle_used"]).all()
assert (qa["scaler_fit_partition"] == "train").all()
assert (~qa["test_used_for_model_selection"]).all()

protocol = pd.DataFrame({
    "פרט": ["טווח נתונים", "חלוקה", "Features", "K", "Seeds", "Covariance", "בחירה", "Shuffle", "Scaling"],
    "ערך": [
        f'{hmm_manifest["config"]["start_date"]} — {hmm_manifest["config"]["end_date"]}',
        "כ-70% Train / 15% Validation / 15% Test, כרונולוגי",
        ", ".join(FEATURES),
        str(hmm_manifest["config"]["k_values"]),
        str(hmm_manifest["config"]["seeds"]),
        hmm_manifest["config"]["covariance_type"],
        "Validation log-likelihood בלבד",
        "לא",
        "Train בלבד",
    ],
})
display(protocol)
'''))

cells.append(md(r'''
ה-assertions מגנים על invariants של הניסוי: אין shuffle של סדרת הזמן, ה-scaler אינו רואה Validation/Test, ו-Test אינו משמש model selection. בכך ההשוואה בין HMM, Baselines ו-Supervised ML נשארת הוגנת.
'''))

cells.append(md(r'''
## 3. Gaussian HMM בקצרה

ה-state החבוי \(S_t\) מתפתח לפי הנחת Markov מסדר ראשון:

\[
P(S_t\mid S_{1:t-1})=P(S_t\mid S_{t-1})
\]

ומטריצת המעבר מכילה \(a_{ij}=P(S_t=j\mid S_{t-1}=i)\). ב-Gaussian HMM:

\[
X_t\mid S_t=k\sim\mathcal{N}(\mu_k,\Sigma_k)
\]

הפרמטרים נלמדים באמצעות Baum-Welch / EM. לכל נכס נבדקו \(K=2,3,4\) וחמישה seeds, והבחירה נעשתה לפי Validation log-likelihood בלבד.

בנוסף ל-hard state נבדק soft posterior, המשתמש בכל \(P(S_t=k\mid X_{1:T})\). כך יום שבו המודל מתלבט בין שני states אינו מטופל כאילו הוא שייך בוודאות ל-state יחיד.
'''))

cells.append(md(r'''
## 4. בחירת K ויציבות בין seeds

Likelihood טוב אינו מספיק: מודל יכול להוסיף states ולהיות רגיש ל-initialization. לכן נבדקה גם יציבות חלוקת הימים באמצעות `Adjusted Rand Index (ARI)`. ARI קרוב ל-1 פירושו שריצות שונות מייצרות כמעט אותה חלוקה, ללא תלות במספור שרירותי של ה-labels.
'''))

cells.append(code(r'''
asset_status = pd.DataFrame(hmm_manifest["asset_status"])[
    ["asset", "selected_k", "selected_seed", "test_start", "test_end"]
].copy()
seed_stability = pd.read_csv(EXTENDED_RUN / "seed_stability_all_assets.csv")
selected_stability = asset_status.merge(
    seed_stability,
    left_on=["asset", "selected_k"],
    right_on=["asset", "k"],
    how="left",
)[["asset", "selected_k", "selected_seed", "mean_pairwise_ARI", "min_pairwise_ARI", "min_state_occupancy_across_seeds_pct"]]
display(selected_stability.round(3))
display(Image(filename=str(EXTENDED_RUN / "seed_stability_ari.png")))
'''))

cells.append(md(r'''
ב-SPY מתקבל ARI ממוצע של כ-0.995 עבור \(K=4\), ב-QQQ כ-0.960 וב-NVDA כ-0.984. לעומת זאת BTC-USD ו-GLD יציבים פחות ב-\(K=4\). הממצא מחדד trade-off בין expressiveness לבין robustness: יותר states יכולים לתאר מבנה עשיר יותר, אבל לא בכל נכס המבנה העשיר יציב באותה מידה.
'''))

cells.append(md(r'''
## 5. SPY כמקרה מבחן: מה מאפיין כל Hidden State?

אין משמעות מובנית למספרי states. לכן השמות ניתנים רק לאחר בדיקת return, volatility, range, drawdown, occupancy ו-dwell time.
'''))

cells.append(code(r'''
spy_states = pd.read_csv(EXTENDED_RUN / "SPY" / "state_summary_history.csv")
state_labels = {
    0: "Normal / moderate volatility",
    1: "Calm / low volatility",
    2: "Rare stress / crisis-like",
    3: "Persistent high volatility",
}
spy_states["interpretation"] = spy_states["state"].map(state_labels)
cols = [
    "state", "interpretation", "frequency_%", "mean_daily_return_%",
    "volatility_daily_%", "mean_daily_range_%", "mean_drawdown_%",
    "worst_drawdown_%", "avg_duration_days", "mean_posterior_confidence",
]
display(spy_states[cols].round(3))
display(Image(filename=str(EXTENDED_RUN / "SPY" / "test_price_by_regime.png")))
'''))

cells.append(md(r'''
ההפרדה חזקה במיוחד ב-SPY. State 1 הוא שקט יחסית: volatility יומית של כ-0.42% ומשך ממוצע של כ-32 ימים. State 2 נדיר — כ-4.5% מההיסטוריה — אך בעל תשואה יומית ממוצעת שלילית, volatility של כ-3.41%, daily range של כ-3.61% ו-mean drawdown של כ-15%-.

לכן `crisis-like` הוא label תיאורי post-hoc הנתמך בסטטיסטיקה; הוא אינו ground truth שהוזן למודל.
'''))

cells.append(md(r'''
## 6. Transition Matrix ומשך המשטר

כאשר \(a_{ii}\) גבוה, state נוטה להתמיד. תחת HMM מסדר ראשון משך השהייה המשתמע הוא גאומטרי והממוצע הוא:

\[
E[D_i]=\frac{1}{1-a_{ii}}
\]

בדקנו האם הממוצע הזה שונה משמעותית מה-dwell time המפוענח בפועל.
'''))

cells.append(code(r'''
display(Image(filename=str(EXTENDED_RUN / "SPY" / "transition_matrix.png")))
duration = pd.read_csv(EXTENDED_RUN / "state_duration_diagnostics.csv")
display(duration[duration["asset"] == "SPY"][
    ["state", "self_transition_probability", "hmm_implied_mean_duration_days", "decoded_empirical_mean_duration_days", "empirical_to_implied_duration_ratio"]
].round(3))
'''))

cells.append(md(r'''
ב-SPY משכי השהייה האמפיריים קרובים יחסית לממוצעים הנגזרים מ-\(1/(1-a_{ii})\). לכן לא נמצא mismatch ברור שמצדיק טענה שהנחת duration נכשלה. הרחבה ל-Hidden Semi-Markov Model יכולה עדיין לבדוק את כל התפלגות ה-duration ולא רק את הממוצע.
'''))

cells.append(md(r'''
## 7. Posterior uncertainty — מידע מעבר ל-label

חושבה entropy מנורמלת:

\[
H_t=-\frac{\sum_k\gamma_t(k)\log\gamma_t(k)}{\log K}
\]

ערך נמוך מצביע על posterior חד; ערך גבוה מצביע על ambiguity בין states.
'''))

cells.append(code(r'''
with (EXTENDED_RUN / "SPY" / "selected_model_summary.json").open(encoding="utf-8") as f:
    spy_model_summary = json.load(f)
unc = spy_model_summary["posterior_uncertainty"]
unc_table = pd.DataFrame({
    "מדד": ["Mean confidence", "Mean entropy", "Entropy on switch days", "Entropy on non-switch days", "Entropy near switch ±1", "Corr entropy vs |return|", "Corr entropy vs rolling vol"],
    "ערך": [unc["mean_confidence"], unc["mean_entropy"], unc["mean_entropy_on_switch_days"], unc["mean_entropy_on_non_switch_days"], unc["mean_entropy_near_switch_plus_minus_1"], unc["entropy_abs_return_correlation"], unc["entropy_rolling_volatility_correlation"]],
})
display(unc_table.round(3))
display(Image(filename=str(EXTENDED_RUN / "SPY" / "posterior_entropy_timeline.png")))
'''))

cells.append(md(r'''
זהו אחד הממצאים החזקים בעבודה: posterior confidence ממוצע ב-SPY הוא כ-96.6%, אבל entropy בימי switch עולה לכ-0.307 לעומת כ-0.054 בימים ללא switch. המתאם שלה עם absolute return או rolling volatility כמעט אפסי. לכן entropy אינה פשוט proxy לתנודתיות; היא מתארת בעיקר **אי-ודאות לגבי גבול בין regimes**.
'''))

cells.append(md(r'''
## 8. External validation מול VIX

`VIX` לא שימש Feature ולא השתתף באימון. לכן הוא מאפשר בדיקת post-hoc חיצונית של פרשנות ה-states.
'''))

cells.append(code(r'''
vix = pd.read_csv(EXTENDED_RUN / "vix_by_spy_regime.csv")
display(vix.round(3))
display(Image(filename=str(EXTENDED_RUN / "vix_by_spy_regime.png")))
with (EXTENDED_RUN / "vix_external_validation.json").open(encoding="utf-8") as f:
    vix_validation = json.load(f)
display(pd.DataFrame([vix_validation]).round(3))
'''))

cells.append(md(r'''
בחלון המבחן State 1 קשור ל-VIX ממוצע של כ-12.96, State 0 לכ-15.34 ו-State 3 לכ-18.55. ב-State 3 כ-26% מהימים היו מעל VIX=20. State 2 הנדיר לא הופיע בחפיפה ולכן אינו מאומת כאן. גם entropy עצמה כמעט אינה מתואמת עם VIX, ולכן אין לזהות בין uncertainty של ה-HMM לבין "פחד בשוק".
'''))

cells.append(md(r'''
## 9. Cross-asset analysis: האם state של SPY הוא מצב שוק רחב יותר?

נבדקה התנהגות יתר הנכסים מותנית ב-state של SPY, תוך שימוש בתאריכים חופפים בלבד. המטרה היא לראות האם return, volatility והקורלציה עם SPY משתנים בין regimes.
'''))

cells.append(code(r'''
cross_asset = pd.read_csv(EXTENDED_RUN / "cross_asset_by_reference_state.csv")
focus_assets = ["QQQ", "IWM", "TLT", "GLD", "HYG", "BTC-USD", "JPM", "NVDA"]
cross_focus = cross_asset[cross_asset["asset"].isin(focus_assets)].copy()
corr_pivot = cross_focus.pivot(index="asset", columns="reference_state", values="correlation_with_reference")
corr_pivot.columns = [f"SPY state {c}" for c in corr_pivot.columns]
display(corr_pivot.round(3))
ax = corr_pivot.plot(kind="bar", figsize=(11, 5))
ax.set_title("Correlation with SPY conditional on SPY hidden state")
ax.set_ylabel("Correlation")
ax.set_xlabel("Asset")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
'''))

cells.append(md(r'''
ב-State 3 הקורלציה עם SPY מגיעה לכ-0.98 ב-QQQ, 0.77 ב-HYG, 0.63 ב-BTC-USD, 0.68 ב-JPM ו-0.77 ב-NVDA. לעומת זאת TLT נמצא באותו state בקורלציה קרובה לאפס ואף מעט שלילית. מכאן עולה שה-HMM אינו רק "צובע" את SPY; חלק מה-states מתלווים גם לשינוי במבנה התלות בין asset classes.
'''))

cells.append(md(r'''
## 10. חיזוי יום-קדימה: HMM מול Baselines

`DPA` הוא שיעור הימים שבהם סימן התשואה נחזה נכון. חשוב לזכור ש-Accuracy מעל 50% אינו בהכרח predictive edge: כאשר רוב ימי Test חיוביים, baseline שמנבא את הכיוון הדומיננטי יכול לקבל DPA גבוה יחסית בלי לזהות signal יומי אמיתי.
'''))

cells.append(code(r'''
hmm_metrics = pd.read_csv(EXTENDED_RUN / "test_metrics_all_assets.csv")
models_for_mean = ["Naive - train mean", "Gaussian HMM - hard state", "Gaussian HMM - soft posterior"]
mean_hmm = hmm_metrics[hmm_metrics["model"].isin(models_for_mean)].groupby("model", as_index=False)["DPA_direction_accuracy"].mean()
supervised_summary = pd.read_csv(SUPERVISED_RUN / "supervised_summary.csv").rename(columns={"mean_DPA_direction_accuracy": "DPA_direction_accuracy"})[["model", "DPA_direction_accuracy"]]
mean_comparison = pd.concat([mean_hmm, supervised_summary], ignore_index=True)
mean_comparison["mean_DPA_%"] = 100 * mean_comparison["DPA_direction_accuracy"]
display(mean_comparison[["model", "mean_DPA_%"]].sort_values("mean_DPA_%", ascending=False).round(2))
display(Image(filename=str(SUPERVISED_RUN / "dpa_comparison_all_assets.png")))
'''))

cells.append(md(r'''
בממוצע בין תשעת הנכסים: Naive train-mean משיג כ-54.54% DPA, Logistic Regression כ-54.14%, Random Forest כ-53.27%, HMM soft-posterior כ-53.06%, HistGradientBoosting כ-52.49% ו-HMM hard-state כ-52.06%.

יש הצלחות מקומיות, למשל Random Forest ב-QQQ וב-BTC-USD ו-soft HMM ב-QQQ, אבל אין מודל שמנצח באופן עקבי. לכן אין בסיס לטענה שה-HMM מספק predictive edge יציב.
'''))

cells.append(md(r'''
## 11. Supervised ML Baselines

שלושת המודלים קיבלו בדיוק את אותם ארבעה Features ואותו split כרונולוגי. ה-hyperparameters נקבעו מראש ולא כוונו לפי Test. בנוסף ל-DPA נמדדו Balanced Accuracy, ROC-AUC, Log Loss ו-Brier Score.
'''))

cells.append(code(r'''
sup = pd.read_csv(SUPERVISED_RUN / "supervised_results_all_assets.csv")
summary = sup.groupby("model", as_index=False).agg(
    mean_DPA=("DPA_direction_accuracy", "mean"),
    mean_balanced_accuracy=("balanced_accuracy", "mean"),
    mean_ROC_AUC=("roc_auc", "mean"),
    mean_log_loss=("log_loss", "mean"),
    mean_brier=("brier_score", "mean"),
)
for c in ["mean_DPA", "mean_balanced_accuracy", "mean_ROC_AUC"]:
    summary[c] *= 100
display(summary.round(2))
'''))

cells.append(md(r'''
גם המודלים supervised נשארים ברוב המדדים סביב signal חלש; Balanced Accuracy ו-ROC-AUC קרובים בדרך כלל ל-50%. לכן הקושי ב-next-day forecasting אינו ייחודי ל-HMM. אם המטרה היחידה הייתה classification, אין יתרון ברור להשתמש ב-HMM. התרומה שלו היא states, transition dynamics, persistence ו-posterior uncertainty — מידע שה-classifiers אינם מייצרים באופן טבעי.
'''))

cells.append(md(r'''
## 12. Walk-forward robustness על SPY

נוספו שלושה expanding windows. בכל fold מבוצעים Train → Validation → Test חדשים בסדר כרונולוגי, והמודל נבחר מחדש ללא שימוש ב-Test.
'''))

cells.append(code(r'''
walk = pd.read_csv(EXTENDED_RUN / "walk_forward_results.csv")
walk_display = walk[["fold", "train_end", "validation_end", "test_start", "test_end", "selected_k", "selected_seed", "naive_train_mean_DPA", "gaussian_hmm_hard_state_DPA", "gaussian_hmm_soft_posterior_DPA"]].copy()
for c in ["naive_train_mean_DPA", "gaussian_hmm_hard_state_DPA", "gaussian_hmm_soft_posterior_DPA"]:
    walk_display[c] *= 100
display(walk_display.round(2))
display(Image(filename=str(EXTENDED_RUN / "walk_forward_dpa.png")))
'''))

cells.append(md(r'''
בשלושת החלונות soft HMM משיג בקירוב 50.36%, 51.82% ו-60.22%, לעומת 48.54%, 52.92% ו-59.85% ל-Naive. הממוצע מעט גבוה יותר ל-soft HMM, אך הוא אינו מנצח בכל fold והפער קטן. לכן ייתכן שמידע על regime מסייע בתקופות מסוימות, אך אין ראיה ליתרון חיזוי יציב.
'''))

cells.append(md(r'''
## 13. מסקנה מרכזית

הניסוי מפריד בין **forecasting** לבין **representation**:

1. HMM אינו מנצח באופן עקבי את Baselines או Supervised ML בחיזוי היום הבא.
2. ב-SPY מתקבלים states נבדלים מאוד ב-return, volatility, range, drawdown ו-duration.
3. מבנה K=4 של SPY יציב מאוד בין seeds.
4. posterior entropy עולה סביב state switches אך אינה פשוט proxy ל-volatility.
5. VIX שלא שימש באימון מפריד בין חלק מה-states.
6. הקורלציות וה-volatility של נכסים אחרים משתנות כתלות ב-SPY regime.

לכן בפרויקט זה Gaussian HMM שימושי יותר כ-**latent market-state estimator** מאשר כ-point-forecasting model ליום הבא.
'''))

cells.append(md(r'''
## 14. Future Work: Regime-aware Reinforcement Learning

המשך טבעי הוא לעבור מהסקת מצב לקבלת החלטה סדרתית. Observation עתידי יכול לכלול את ארבעת ה-Features, posterior probabilities של HMM, posterior entropy והפוזיציה הנוכחית. Action יכול להיות `sell / hold / buy` או משקל רציף בתיק, וה-reward צריך להיות תשואה נטו לאחר transaction costs עם penalty ל-turnover/סיכון.

הניסוי הנקי ביותר יהיה ablation:

- RL עם Features בלבד;
- RL עם Features + HMM posterior;
- RL עם Features + posterior + entropy.

כך ניתן לבדוק האם מידע לטנטי על regime מוסיף ערך אינקרמנטלי למדיניות. לא בוצע כאן ניסוי RL, ולכן אין טענה שהכיוון הזה ישפר ביצועים; הוא דורש walk-forward, multiple seeds ועלויות עסקה מפורשות.
'''))

cells.append(md(r'''
## 15. שחזור והרצה

המחברת מציגה את התוצאות מה-artifacts המאומתים ואינה דורשת retraining. כדי ליצור ריצות חדשות:

```bash
python run_extended_analysis.py
python analyze_extended_context.py --run-dir experiments_extended/<run_id>
python run_supervised_baselines.py
```

כל ריצה חדשה נשמרת בתיקייה timestamped חדשה כדי לא לדרוס את התוצאות שעליהן מבוסס הדוח.
'''))

nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)

nbf.write(nb, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
