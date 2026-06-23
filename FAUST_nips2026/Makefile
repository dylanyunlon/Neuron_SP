# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Makefile for DES-LOC NeurIPS 2026 paper

TEX      = pdflatex
BIBTEX   = bibtex
MAIN     = main
TEXFLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: pdf clean

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex neurips_2026.sty
	$(TEX) $(TEXFLAGS) $(MAIN)
	$(TEX) $(TEXFLAGS) $(MAIN)

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).out $(MAIN).nav $(MAIN).snm $(MAIN).toc $(MAIN).fls \
	      $(MAIN).fdb_latexmk $(MAIN).synctex.gz
