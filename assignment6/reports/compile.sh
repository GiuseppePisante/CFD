#!/bin/bash
mkdir -p aux_files
pdflatex -output-directory=aux_files CFD1-Exercise6-Pampalone_Pisante_Raffaelli.tex
mv aux_files/CFD1-Exercise6-Pampalone_Pisante_Raffaelli.pdf .