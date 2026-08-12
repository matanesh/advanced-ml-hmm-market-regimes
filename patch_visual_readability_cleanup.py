from pathlib import Path

# 1) Keep the table of contents on one page instead of leaving a nearly empty
#    continuation page for the appendix entry.
report = Path("reports/report.tex")
text = report.read_text(encoding="utf-8")
old = "\\begingroup\n\\small\n\\setlength{\\parskip}{0pt}\n\\tableofcontents"
new = "\\begingroup\n\\footnotesize\n\\setstretch{1.0}\n\\setlength{\\parskip}{0pt}\n\\tableofcontents"
if old not in text:
    raise RuntimeError("TOC block not found")
report.write_text(text.replace(old, new, 1), encoding="utf-8")

# 2) Force negative numeric cells in the Hebrew state-summary table into math
#    direction so the minus sign appears before the number visually.
states = Path("reports/sections/05_regime_results.tex")
text = states.read_text(encoding="utf-8")
replacements = {
    "& -1.72 &": "& \\(-1.72\\) &",
    "& -0.42 &": "& \\(-0.42\\) &",
    "& -0.265 &": "& \\(-0.265\\) &",
    "& -15.28 &": "& \\(-15.28\\) &",
    "& -8.11 &": "& \\(-8.11\\) &",
}
for old, new in replacements.items():
    if old not in text:
        raise RuntimeError(f"State-table value not found: {old}")
    text = text.replace(old, new, 1)
states.write_text(text, encoding="utf-8")

# 3) Avoid an awkward mixed-direction Bull/Bear heading. Keep the English
#    terminology in a compact parenthetical after the Hebrew explanation.
discussion = Path("reports/sections/07_discussion.tex")
text = discussion.read_text(encoding="utf-8")
old = "\\subsection{מדוע לא קראנו למצבים \\tech{Bull} ו־\\tech{Bear}?}"
new = "\\subsection{מדוע לא קראנו למצבים שוק שורי ושוק דובי (\\tech{Bull/Bear})?}"
if old not in text:
    raise RuntimeError("Bull/Bear heading not found")
discussion.write_text(text.replace(old, new, 1), encoding="utf-8")

print("Applied final visual readability cleanup.")
