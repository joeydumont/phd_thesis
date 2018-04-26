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
LIGHT_TABLE=light_table

# -- Figure generation options (TeX)
FIG_INPUTDIR=light_table
FIG_LATEX=latexmk -xelatex
FIG_OUTDIR=figs
FIG_LATEXFLAGS=--outdir=$(FIG_OUTDIR)/

# -- Figure generation (Python)
PY_INPUTDIR=light_table
PY_BINARY=python
PY_OUTDIR=figs
PY_FLAGS=
BIN_FOLDER=bin

# -- User targets.
default: $(OUTDIR)/$(PROJECT).pdf

display: default
	(${PDFVIEWER} $(OUTDIR)/$(PROJECT).pdf &)

### Extra Targets
# Removes the mendeley tags, groups and file metadata out of the bibliographic
# files.
prune-bib: $(BIB_FILES)
	$(foreach file,$(BIB_FILES), $(shell sed -i.bak '/^file\|^mendeley\|keywords/d' $(file)))
	$(foreach file,$(BIB_FILES), $(shell biber -q -O $(file) --output-format=bibtex --configfile=biber-tool.conf --tool $(file)))
	$(foreach file,$(BIB_FILES), $(shell sed -i 's/USERA/YEAR/' $(file)))

log: $(OUTDIR)/$(PROJECT).pdf
	pplatex -i $(OUTDIR)/$(PROJECT).log

### Compilation Flags
PDFLATEX_FLAGS  = -halt-on-error -output-directory $(OUTDIR)/

TEXINPUTS = .:$(OUTDIR)/
TEXMFOUTPUT = $(OUTDIR)/

### File Types (for dependencies)
TEX_FILES       = $(shell find . -name '*.tex' -or -name '*.sty' -or -name '*.cls')
BIB_FILES       = $(shell find . -name '*.bib')
BST_FILES       = $(shell find . -name '*.bst')

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
	rm -rf $(PY_OUTDIR)
	rm -rf $(FIG_OUTDIR)

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

# -------------------- Generation of LaTeX based figures -------------------- #
LATEX_IMAGES_DEPS := $(LIGHT_TABLE)/DomainDecomposition/hpc-domaindecomposition.tex
LATEX_IMAGES_DEPS += $(LIGHT_TABLE)/ParabolicMirrors/parabola_hna.tex
LATEX_IMAGES_DEPS += $(LIGHT_TABLE)/ParabolicMirrors/parabola_vsf.tex
LATEX_IMAGES_DEPS += $(LIGHT_TABLE)/scatteringSystem/scatteringSystem.tex

latex_images: $(LATEX_IMAGES_DEPS) | $(FIG_OUTDIR)/
	$(foreach file, $^, $(shell $(FIG_LATEX) $(FIG_LATEXFLAGS) $(file)))

# ------------------- Generation of Python based figures -------------------- #
PYTHON_IMAGES_DEPS := $(LIGHT_TABLE)/ClassicalLimit/ClassicalLimit.py
PYTHON_IMAGES_DEPS += $(LIGHT_TABLE)/ConvergenceAnalysis/ConvergenceAnalysis.py
PYTHON_IMAGES_DEPS += $(LIGHT_TABLE)/ParallelEfficiency/ParallelEfficiency.py
PYTHON_IMAGES_DEPS += $(LIGHT_TABLE)/RichardsWolf/FastRW_plot.py
PYTHON_IMAGES_DEPS += $(LIGHT_TABLE)/SCIntegrandOscillation/IntegrandOscillation.py
PYTHON_IMAGES_DEPS += $(LIGHT_TABLE)/TightlyFocusedFields/Ellipticity.py
PYTHON_IMAGES_CMD   = $(cd $$(readlink -f $(file)) && $(PY_BINARY) $$(basename $(file)) && cd $$(basename $(PROJECT).TEX))
python_images: $(PYTHON_IMAGES_DEPS) | $(FIG_OUTDIR)/
	# -- Generate the figures.
	$(foreach file, $^, $(PYTHON_IMAGES_CMD))

	# -- Manually copy them over.
	cp $(LIGHT_TABLE)/ClassicalLimit/ClassicalLimit.pdf                                          $(FIG_OUTDIR)/
	cp $(LIGHT_TABLE)/ConvergenceAnalysis/ConvergenceAll.pdf                                     $(FIG_OUTDIR)/
	cp $(LIGHT_TABLE)/IntensityHistory/figs/IntensityHistory.pdf                                 $(FIG_OUTDIR)/ # No way to automate creation of this fig.
	cp $(LIGHT_TABLE)/ParallelEfficiency/ParallelEfficiency*                                     $(FIG_OUTDIR)/
	cp $(BIN_FOLDER)/Fields/RichardsWolf/f0.04375/RichardsWolf_fp.pdf                            $(FIG_OUTDIR)/RichardsWolf_fpNA1.pdf
	cp $(BIN_FOLDER)/Fields/Stratto/na_vsf_lin_g/stra-vsf-lin_focal0.04375.BQ/StrattonChu_fp.pdf $(FIG_OUTDIR)/StrattonChu_fpNA1.pdf
	cp $(BIN_FOLDER)/Fields/RichardsWolf/f0.00875/RichardsWolf_fp.pdf                            $(FIG_OUTDIR)/RichardsWolf_fpVSF.pdf
	cp $(BIN_FOLDER)/Fields/Stratto/na_vsf_lin_g/stra-vsf-lin_focal0.00875.BQ/StrattonChu_fp.pdf $(FIG_OUTDIR)/StrattonChu_fpVSF.pdf
	cp $(BIN_FOLDER)/Fields/Stratto/na_vsf_lin_g/stra-vsf-lin_00001.BQ/00001.BQ/ElectricIntensityTimeWaterfall-cropped.pdf                                                $(FIG_OUTDIR)/ElectricIntensityTimeWaterfallf0.007.pdf
	cp $(BIN_FOLDER)/Fields/Stratto/na_vsf_lin_g/stra-vsf-lin_00010.BQ/00010.BQ/ElectricIntensityTimeWaterfall-cropped.pdf                                                $(FIG_OUTDIR)/ElectricIntensityTimeWaterfallf0.04375.pdf
	cp $(LIGHT_TABLE)/SCIntegrandOscillation/phase*.pdf                                          $(FIG_OUTDIR)/

#%.pdf: %.svg | $(FIG_OUTDIR)/
#	inkscape --file=$< --export-area-drawing --without-gui --export-pdf=$(FIG_OUTDIR)/$(notdir $@)
#	$(foreach file, $(INKSCAPE_FILES), $(shell inkscape --file=$(file) --export-area-drawing --without-gui --export-pdf=$(basename $(notdir $(file))).pdf))

# ------------------- Generation of the PDF of the thesis. ------------------ #
$(OUTDIR)/:
	mkdir -p $(OUTDIR)/

$(OUTDIR)/$(PROJECT).aux: $(TEX_FILES) | class $(OUTDIR)/
	$(LATEX) $(PDFLATEX_FLAGS) $(PROJECT)

$(OUTDIR)/$(PROJECT).bbl: $(BIB_FILES) | $(OUTDIR)/$(PROJECT).aux
	bibtex $(OUTDIR)/$(PROJECT)
	$(LATEX) $(PDFLATEX_FLAGS) $(PROJECT)

$(OUTDIR)/$(PROJECT).pdf: $(OUTDIR)/$(PROJECT).aux $(if $(BIB_FILES), $(OUTDIR)/$(PROJECT).bbl)
	$(LATEX) $(PDFLATEX_FLAGS) $(PROJECT).tex
	cp $@ .
