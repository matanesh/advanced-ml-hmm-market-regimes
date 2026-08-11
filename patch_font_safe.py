from pathlib import Path

path = Path("reports/report.tex")
text = path.read_text(encoding="utf-8")
start_marker = "% Prefer a restrained serif body similar to a traditional academic paper."
end_marker = "\n\\setmonofont{DejaVu Sans Mono}"
start = text.index(start_marker)
end = text.index(end_marker, start)
new_block = r'''% Readability-first typography with one robust Hebrew/Latin family.
% DejaVu Sans has complete coverage in the build environment and keeps mixed
% Hebrew/English text stable while providing a clearer long-form reading face.
\setmainfont{DejaVu Sans}
\setsansfont{DejaVu Sans}
\newfontfamily\hebrewfont{DejaVu Sans}[Script=Hebrew,Renderer=HarfBuzz]
\newfontfamily\hebrewfontsf{DejaVu Sans}[Script=Hebrew,Renderer=HarfBuzz]
\newfontfamily\englishfont{DejaVu Sans}[Renderer=HarfBuzz]'''
text = text[:start] + new_block + text[end:]
path.write_text(text, encoding="utf-8")
print("Applied DejaVu Sans readability font patch only.")
