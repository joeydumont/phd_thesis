# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created       Mar. 1st, 2018                                                #
# Description:  We verify that the StrattoCalculator fields obey Maxwell's    #
#               equations.                                                    #
# Dependencies: - NumPy                                                       #
#               - SciPy                                                       #
#               - Matplotlib                                                  #
#               - h5py                                                        #
# --------------------------------------------------------------------------- #

# --------------------------- Modules Importation --------------------------- #
# -- Plotting
import matplotlib as mpl
mpl.use('pgf')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches

# -- Analysis
import numpy as np
import scipy.constants as cst
import math
import h5py
from scipy import interpolate

# -- OS and other stuff.
import time
import argparse
import imp

# -- Load our custom module.
vphys = imp.load_source('vphys', "../../python-tools/vphys.py")
AnalysisStratto = imp.load_source('AnalysisStratto', "../../python-tools/AnalysisStrattoCalculator.py")
import vphys
import AnalysisStratto as analstrat


# ------------------------------ Configuration ------------------------------ #

#-- We reset the LaTeX parameters to enable XeLaTeX.
mpl.rcParams.update(vphys.default_pgf_configuration())


# ----------------------------- Argument Parsing ---------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--output-directory",
                      type=str,
                      default="figs/",
                      help="Directory where the figures will be saved.")
args = parser.parse_args()

# ------------------------------ MAIN FUNCTION ------------------------------ #

# -- Open the file for analysis.
analysis_object = analstrat.Analysis3D(freq_field="../../bin/MaxwellVerification/Field_reflected.hdf5")

# -- Evaluate the surface integral at specified values of r.

# -- Evaluate the line integral at specified values of r.

# -- Evaluate the line integrals at specified values of r.
