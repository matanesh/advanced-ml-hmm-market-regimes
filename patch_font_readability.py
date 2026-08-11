from pathlib import Path

path = Path("reports/report.tex")
text = path.read_text(encoding="utf-8")
old = r'''% Prefer a restrained serif body similar to a traditional academic paper.
% FreeSerif/FreeSans cover both Hebrew and Latin, which keeps mixed-language
% sentences stable. The previous DejaVu setup remains the portable fallback.
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
}'''
new = r'''% Readability-first typography for Hebrew body text.
% Noto Serif Hebrew is cleaner and more open than FreeSerif for long-form reading.
% Sans-serif headings remain restrained and modern. Portable fallbacks are kept.
\IfFontExistsTF{Noto Serif Hebrew}{
  \setmainfont{Noto Serif}
  \setsansfont{Noto Sans}
  \newfontfamily\hebrewfont{Noto Serif Hebrew}[Script=Hebrew,Renderer=HarfBuzz]
  \newfontfamily\hebrewfontsf{Noto Sans Hebrew}[Script=Hebrew,Renderer=HarfBuzz]
  \newfontfamily\englishfont{Noto Serif}[Renderer=HarfBuzz]
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
if text.count(old) != 1:
    raise RuntimeError(f"Expected exactly one font block, found {text.count(old)}")
path.write_text(text.replace(old, new), encoding="utf-8")
print("Font block replaced with Noto Serif Hebrew / Noto Sans Hebrew.")
