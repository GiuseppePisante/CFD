#!/bin/bash
mkdir -p aux_files
pdflatex -output-directory=aux_files temp.tex
mv aux_files/temp.pdf .