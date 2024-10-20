#!/bin/bash
mkdir -p aux_files
pdflatex -output-directory=aux_files week12.tex
mv aux_files/week12.pdf .