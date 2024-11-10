#!/bin/bash
mkdir -p aux_files
pdflatex -output-directory=aux_files temp_marti.tex
mv aux_files/temp_marti.pdf .