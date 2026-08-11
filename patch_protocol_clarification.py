"""Clarify final fitting-history details in the generated submission notebook.

Run after ``build_final_notebook.py`` and ``patch_final_notebook.py`` and before
executing the notebook.  This is wording/source-only QA: it does not alter any
experimental artifact or numerical result.
"""

from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).resolve().parent / "HMM_Market_Regimes_Project.ipynb"


TEXT_REPLACEMENTS = {
    "הנתונים התבקשו לטווח `2014-01-01` עד `2024-12-31`. זהו חלון הבקשה המשותף; זמינות התצפיות בפועל שונה מעט בין נכסים, ובפרט `BTC-USD` מתחיל מאוחר יותר. כל split מחושב מתוך התצפיות הזמינות בפועל לכל נכס. החלוקה כרונולוגית: כ-70% Train, כ-15% Validation וכ-15% Test. שורות גבול שבהן יעד היום הבא חוצה partition הוסרו, וכל scaling מותאם על Train בלבד.":
    "הנתונים התבקשו לטווח `2014-01-01` עד `2024-12-31`. זהו חלון הבקשה המשותף; זמינות התצפיות בפועל שונה מעט בין נכסים, ובפרט `BTC-USD` מתחיל מאוחר יותר. כל split מחושב מתוך התצפיות הזמינות בפועל לכל נכס. החלוקה כרונולוגית: כ-70% Train, כ-15% Validation וכ-15% Test, ושורות גבול שבהן יעד היום הבא חוצה partition הוסרו. בשלב בחירת ה-HMM ה-scaler מותאם על Train בלבד; לאחר נעילת K וה-seed מתבצע refit של ה-HMM וה-scaler על Train+Validation לפני ההערכה על Test. מודלי ה-Supervised הקבועים מאומנים על Train בלבד.",

    "ה-assertions מגנים על invariants של הניסוי: אין shuffle של סדרת הזמן, ה-scaler אינו רואה Validation/Test, ו-Test אינו משמש model selection. בכך ההשוואה בין HMM, Baselines ו-Supervised ML נשארת הוגנת.":
    "ה-assertions בתא מתייחסים לריצת ה-Supervised: אין shuffle, ה-scaler של Logistic Regression מותאם על Train בלבד, ו-Test אינו משמש model selection. ב-HMM נשמר אותו עיקרון בזמן בחירת המועמדים, אך לאחר בחירת K וה-seed המודל הסופי עובר refit על Train+Validation. לכן Test נשאר נעול בשני המסלולים, אך כמות ה-pre-Test data ששימשה להתאמה הסופית אינה זהה לחלוטין.",

    "שלושת המודלים קיבלו בדיוק את אותם ארבעה Features ואותו split כרונולוגי. ה-hyperparameters נקבעו מראש ולא כוונו לפי Test. בנוסף ל-DPA נמדדו Balanced Accuracy, ROC-AUC, Log Loss ו-Brier Score.":
    "שלושת המודלים קיבלו בדיוק את אותם ארבעה Features ואותו split כרונולוגי. ה-hyperparameters נקבעו מראש ולא כוונו לפי Validation או Test, והמודלים אומנו על Train בלבד. לעומת זאת, ה-HMM הסופי עובר refit על Train+Validation לאחר model selection. לכן זו השוואת Baselines משלימה על אותו Test ולא תחרות שבה לכל המודלים ניתנה בדיוק אותה כמות נתוני אימון. בנוסף ל-DPA נמדדו Balanced Accuracy, ROC-AUC, Log Loss ו-Brier Score.",

    "יש הצלחות מקומיות, למשל Random Forest ב-QQQ וב-BTC-USD ו-soft HMM ב-QQQ, אבל אין מודל שמנצח באופן עקבי. הצגת כל ה-Baselines חשובה במיוחד משום ש-Discrete Markov Chain הפשוט נמצא בממוצע מעל שני ה-HMM variants. לכן אין בסיס לטענה שה-HMM מספק predictive edge יציב.":
    "יש הצלחות מקומיות, למשל Random Forest ב-QQQ וב-BTC-USD ו-soft HMM ב-QQQ, אבל אין מודל שמנצח באופן עקבי. הצגת כל ה-Baselines חשובה במיוחד משום ש-Discrete Markov Chain הפשוט נמצא בממוצע מעל שני ה-HMM variants. מאחר שמודלי ה-Supervised אומנו על Train בלבד בעוד ה-HMM הסופי הותאם מחדש על Train+Validation, אין לפרש את הדירוג הממוצע כתחרות מובהקת ביניהם. בכל מקרה, אין בסיס לטענה שה-HMM מספק predictive edge יציב.",
}


def main() -> None:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)

    nb = nbformat.read(NOTEBOOK, as_version=4)
    counts = {old: 0 for old in TEXT_REPLACEMENTS}

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            for old, new in TEXT_REPLACEMENTS.items():
                if old in cell.source:
                    cell.source = cell.source.replace(old, new)
                    counts[old] += 1

        if cell.cell_type == "code" and "protocol = pd.DataFrame" in cell.source:
            old = '        "Train בלבד",\n'
            new = '        "Train בבחירת HMM; Train+Validation ב-refit HMM; Train בלבד ב-Supervised",\n'
            if old not in cell.source:
                raise RuntimeError("Could not find protocol-table scaling row")
            cell.source = cell.source.replace(old, new, 1)

    bad = {old[:80]: count for old, count in counts.items() if count != 1}
    if bad:
        raise RuntimeError(f"Protocol clarification replacements did not match exactly once: {bad}")

    nbformat.write(nb, NOTEBOOK)
    print(f"Patched {NOTEBOOK.name}: {len(TEXT_REPLACEMENTS)} protocol clarifications")


if __name__ == "__main__":
    main()
