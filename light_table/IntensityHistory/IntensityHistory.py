# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created       Feb. 15th, 2018                                               #
# Description:  We recreate Mourou's figure from:                             #
#               G. Mourou, J. A. Wheeler, and T. Tajima, "Extreme light,"     #
#                 Europhys. News 46, 31–35 (2015).                            #
# Dependencies: - NumPy                                                       #
#               - Matplotlib                                                  #
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
from scipy import interpolate

# -- OS and other stuff.
import time
import argparse

# ------------------------------ Configuration ------------------------------ #
#-- We reset the LaTeX parameters to enable XeLaTeX.
pgf_with_pdflatex = {
   "font.family": "serif", # use serif/main font for text elements
   "text.usetex": True,    # use inline math for ticks
   "pgf.rcfonts": False,   # don't setup fonts from rc parameters
   "pgf.preamble": [
        r"\usepackage{amsmath}",
        r"\usepackage{siunitx}",
        r"\usepackage{mathspec}",
        r"\usepackage[charter]{mathdesign}",
        r"\usepackage{fontspec}",
        r"\setmathfont{Fira Sans}",
        r"\setmainfont{Oswald}",
        ]
}
mpl.rcParams.update(pgf_with_pdflatex)
mpl.rcParams['font.size'] = 10
mpl.rcParams['grid.linestyle'] = '--'

# -------------------------------- Functions -------------------------------- #
def mkdir_p(mypath):
  """
  Creates a directory. equivalent to using mkdir -p on the command line
  """

  from errno import EEXIST
  from os import makedirs,path

  try:
      makedirs(mypath)
  except OSError as exc: # Python >2.5
      if exc.errno == EEXIST and path.isdir(mypath):
          pass
      else: raise


def adjust_spines(ax, spines):
  """

  """
  for loc, spine in ax.spines.items():
      if loc in spines:
          spine.set_position(('outward', 10))  # outward by 10 points
          #spine.set_smart_bounds(True)
          spine.set_color('gray')
      else:
          spine.set_color('none')  # don't draw spine

  # turn off ticks where there is no spine
  if 'left' in spines:
      ax.yaxis.set_ticks_position('left')
  else:
      # no yaxis ticks
      ax.yaxis.set_ticks([])

  if 'bottom' in spines:
      ax.xaxis.set_ticks_position('bottom')
  else:
      # no xaxis ticks
      ax.xaxis.set_ticks([])

# ------------------------------ MAIN FUNCTION ------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument("--output-directory",
                      type=str,
                      default="figs/",
                      help="Directory where the figures will be saved.")
args = parser.parse_args()

# ----------------------------------- DATA ---------------------------------- #

intensities_by_year = np.zeros((0,2))

intensities_by_year = np.append(intensities_by_year,[[1960,1e10]], axis=0) # From og figure, verify.
intensities_by_year = np.append(intensities_by_year,[[1965,1e14]], axis=0) # From og figure, verify
intensities_by_year = np.append(intensities_by_year,[[1988,1e15]], axis=0) # From og figure, verify
intensities_by_year = np.append(intensities_by_year,[[1990,1e18]], axis=0) # SLAC
intensities_by_year = np.append(intensities_by_year,[[2002,1e22]], axis=0) # CUOS
intensities_by_year = np.append(intensities_by_year,[[2018,1e24]], axis=0) # INRS
intensities_by_year = np.append(intensities_by_year,[[2020,1e25]], axis=0) # ELI



# ----------------------------------- Plot ---------------------------------- #

# -- We create the figure.
figIHistory = plt.figure(figsize=(6,3.375))
axIHistory  = figIHistory.add_subplot(111)
adjust_spines(axIHistory, ['left', 'bottom'])

# -- Plot the data.
axIHistory.plot(intensities_by_year[:,0], intensities_by_year[:,1])

# -- Y axis decorations
axIHistory.set_yscale('log')
axIHistory.set_ylim((1e8,1e32))
axIHistory.set_yticks(([1e10,1e15,1e20,1e25,1e30]))
axIHistory.tick_params(axis='y', colors='gray')
axIHistory.set_ylabel("Focused Intensity\n"+r"(\si{\watt\per\cm\squared})", color='gray', rotation='horizontal', ha='left')
axIHistory.yaxis.set_label_coords(-0.15,1)

# -- X axis decorations
axIHistory.set_xlim((1955,2035))
axIHistory.tick_params(axis='x', colors='gray')
axIHistory.set_xlabel('Year', color='gray')


# -- Add regions delimited by transparent rectangles.
plt.pcolor([[1955,1e8], [1955,1e8]], cmap=plt.cm.Greens)


mkdir_p(args.output_directory)
plt.savefig(args.output_directory+"IntensityHistory.pdf", bbox_inches='tight')

# -- We remove any superfluous decoration.
# Remove the axis decorations on the right and on the top.
#axPres.spines['top'].set_visible(False)
#axPres.spines['right'].set_visible(False)

# Make the remaining spines a light gray.
#axPres.spines['bottom'].set_color('gray')
#axPres.spines['left'].set_color('gray')


# # -- Set the x ticks.
# axPres.set_xscale('log')
# axPres.set_xlim((0.75,500))
# axPres.set_xticks((nb_procs))
# axPres.set_xticklabels( (r'1', r'2', r'4', r'12', r'24', r'48', r'96', r'192', r'384'), color='gray' )
# axPres.xaxis.set_ticks_position('bottom')

# for tic in axPres.xaxis.get_major_ticks():
#   tic.tick1On = tic.tick2On = False

# # -- Set the y ticks.
# axPres.set_ylim((0,1))
# axPres.set_yticks((0.0,0.5,1.0))
# axPres.set_yticklabels((r'0', '', r'1'))
# axPres.yaxis.set_ticks_position('left')
# axPres.tick_params(axis='y', colors='gray')

# #for tac in axPres.yaxis.get_major_ticks():
# #  tac.tick1On = tac.tick2On = False
# for toc in axPres.xaxis.get_minor_ticks():
#   toc.tick1On = toc.tick2On = False

# # -- Set the titles of the axes.
# axPres.set_ylabel(r"Efficacit\'e", color='gray', rotation='horizontal')
# axPres.yaxis.set_label_position('right')
# axPres.set_xlabel(r"Nombre de processeurs", color='gray')

plt.show()