### latex.makefile
# Author: Joey Dumont
# Based on latex.makefile by: Jason Hiebel
# <https://github.com/JasonHiebel/latex.makefile>

# This is a simple makefile for compiling LaTeX documents. The core assumption
# is that the resulting documents should have any parameters effecting
# rendering quality set to theoretical limits and that all fonts should be
# embedded. While typically overkill, the detriment to doing so is negligible.

# Targets:
#    default : compiles the document to a PDF file using the defined
#              latex generating engine. (pdflatex, xelatex, etc)
#    display : displays the compiled document in a common PDF viewer.
#              (currently linux = evince, OSX = open)
#    clean   : removes the $(OUTDIR)/ directory holding temporary files
#              and the $(OUTDIR_FIGS)/ directory holding the binary figures.

# -- Main project options.
PROJECT=phd_thesis_jayd
CLASS=inrsthesis
LATEX=pdflatex
OUTDIR=out

# -- Figure generation options (TeX)
FIG_INPUTDIR=light_table
FIG_LATEX=pdflatex
FIG_OUTDIR=figs
FIG_LATEXFLAGS=-output-directory $(FIG_OUTDIR)/

# -- Figure generation (Python)
PY_INPUTDIR=light_table
PY_BINARY=python
PY_OUTDIR=figs


default: $(OUTDIR)/$(PROJECT).pdf

display: default
	(${PDFVIEWER} $(OUTDIR)/$(PROJECT).pdf &)


### Compilation Flags
PDFLATEX_FLAGS  = -halt-on-error -output-directory $(OUTDIR)/

TEXINPUTS = .:$(OUTDIR)/
TEXMFOUTPUT = $(OUTDIR)/

### File Types (for dependencies)
TEX_FILES       = $(shell find . -name '*.tex' -or -name '*.sty' -or -name '*.cls')
BIB_FILES       = $(shell find . -name '*.bib')
BIB_FILES_RAW   = $(shell find . -name '*.bib.raw')
BST_FILES       = $(shell find . -name '*.bst')
IMG_FILES       = $(shell find . -path '*.jpg' -or -path '*.png' -or \( \! -path './$(OUTDIR)/*.pdf' -path '*.pdf' \) )
TEX_IMAGE_FILES = $(shell find ./$(FIG_INPUTDIR)/ -name '*.tex')


### Standard PDF Viewers
# Defines a set of standard PDF viewer tools to use when displaying the result
# with the display target. Currently chosen are defaults which should work on
# most linux systems with GNOME installed and on all OSX systems.

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
PDFVIEWER = okular
endif

ifeq ($(UNAME), Darwin)
PDFVIEWER = open
endif

### Clean
# This target cleans the temporary files generated by the TeX programs in
# use. All temporary files generated by this makefile will be placed in $(OUTDIR)/
# so cleanup is easy.

clean::
	rm -rf $(OUTDIR)/

### Core Latex Generation
# Performs the typical build process for latex generations so that all
# references are resolved correctly. If adding components to this run-time
# always take caution and implement the worst case set of commands.
# Example: latex, bibtex, latex, latex
#
# Note the use of order-only prerequisites (prerequisites following the |).
# Order-only prerequisites do not effect the target -- if the order-only
# prerequisite has changed and none of the normal prerequisites have changed
# then this target IS NOT run.
#
# In order to function for projects which use a subset of the provided features
# it is important to verify that optional dependencies exist before calling a
# target; for instance, see how bibliography files (.bbl) are handled as a
# dependency.

class:
	cd $(CLASS); yes | latex  $(CLASS).ins

$(FIG_OUTDIR)/:
	mkdir -p $(FIG_OUTDIR)/

latex_images: $(TEX_IMAGE_FILES) | $(FIG_OUTDIR)/
	$(FIG_LATEX) $(FIG_LATEXFLAGS) $(TEX_IMAGE_FILES)

$(OUTDIR)/:
	mkdir -p $(OUTDIR)/

$(OUTDIR)/$(PROJECT).aux: $(TEX_FILES) $(IMG_FILES) | class $(OUTDIR)/
	$(LATEX) $(PDFLATEX_FLAGS) $(PROJECT)

$(OUTDIR)/$(PROJECT).bbl: $(BIB_FILES) | $(OUTDIR)/$(PROJECT).aux
	bibtex $(OUTDIR)/$(PROJECT)
	$(LATEX) $(PDFLATEX_FLAGS) $(PROJECT)

$(OUTDIR)/$(PROJECT).pdf: $(OUTDIR)/$(PROJECT).aux $(if $(BIB_FILES), $(OUTDIR)/$(PROJECT).bbl) | latex_images
	$(LATEX) $(PDFLATEX_FLAGS) $(PROJECT)
	cp $@ .


### Extra Targets
# Removes the mendeley tags, groups and file metadata out of the bibliographic
# files.
prune-bib: $(BIB_FILES)
	sed -i.bak '/^file\|^mendeley\|keywords/d' $^
	biber --tool $^

log: $(OUTDIR)/$(PROJECT).pdf
	pplatex -i $(OUTDIR)/$(PROJECT).log
