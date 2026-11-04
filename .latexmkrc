# Use XeLaTeX as the PDF engine
$pdf_mode = 5;
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';

# Ensure biber is used for bibliography
$biber = 'biber %O %S';

# Force latexmk to never fall back to pdflatex
$pdflatex = 'echo "ERROR: Use xelatex, not pdflatex" && exit 1';
