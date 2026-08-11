from pathlib import Path

path = Path("reports/report.tex")
text = path.read_text(encoding="utf-8")
start = text.index("% Prefer a restrained serif body similar to a traditional academic paper.")
end = text.index("\n\\setmonofont{DejaVu Sans Mono}", start)
new_block = r'''% Readability-first typography for Hebrew long-form reading.
% David CLM provides a familiar, open Hebrew book face; DejaVu Sans is used
% for headings and explicit English spans. Portable fallbacks are retained.
\IfFontExistsTF{David CLM}{
  \setmainfont{David CLM}
  \setsansfont{DejaVu Sans}
  \newfontfamily\hebrewfont{David CLM}[Script=Hebrew,Renderer=HarfBuzz]
  \newfontfamily\hebrewfontsf{DejaVu Sans}[Script=Hebrew,Renderer=HarfBuzz]
  \newfontfamily\englishfont{DejaVu Serif}[Renderer=HarfBuzz]
}{
  \IfFontExistsTF{FreeSerif}{
    \setmainfont{FreeSerif}
    \setsansfont{FreeSans}
    \newfontfamily\hebrewfont{FreeSerif}[Script=Hebrew,Renderer=HarfBuzz]
    \newfontfamily\hebrewfontsf{FreeSans}[Script=Hebrew,Renderer=HarfBuzz]
    \newfontfamily\englishfont{FreeSerif}[Renderer=HarfBuzz]
  }{
    \setmainfont{DejaVu Sans}
    \setsansfont{DejaVu Sans}
    \newfontfamily\hebrewfont{DejaVu Sans}[Script=Hebrew,Renderer=HarfBuzz]
    \newfontfamily\hebrewfontsf{DejaVu Sans}[Script=Hebrew,Renderer=HarfBuzz]
    \newfontfamily\englishfont{DejaVu Sans}[Renderer=HarfBuzz]
  }
}'''
text = text[:start] + new_block + text[end:]
path.write_text(text, encoding="utf-8")
print("Applied David CLM body-font patch only.")
