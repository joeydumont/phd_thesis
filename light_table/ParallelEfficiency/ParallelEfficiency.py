# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created:      Apr. 4th, 2018                                                #
# Description:  We plot the parallel efficiency of the StrattoCalculator on   #
#               different clusters.                                           #
# Dependencies: - NumPy                                                       #
#               - SciPy                                                       #
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
import matplotlib.ticker as ticker

# -- Analysis
import numpy as np
import scipy.constants as cst
import math
import h5py
from scipy import interpolate
import scipy.integrate as integrate
import scipy.stats as stats

# -- OS and other stuff.
import time
import argparse
import imp

# -- Load our custom modules.
vphys           = imp.load_source('vphys', "../../python-tools/vphys.py")
import vphys


# ------------------------------ Configuration ------------------------------ #
# -- LaTeX

#-- We reset the LaTeX parameters to enable XeLaTeX.
mpl.rcParams.update(vphys.default_pgf_configuration())

# --------------------------- Function Definition --------------------------- #

def ReduceDuplicateEntries(data, numpy_callable=np.mean):
    """
    Find the entries with the same labels (assumed to be the data in the first
    column) and reduce it according to a user-supplied function. By default,
    we take the average of the duplicate entries.
    """
    result_unique_labels  = np.unique(data[:,0])

    ncols = 1
    try:
        ncols = data[:,1].shape[1]
        result_reduced_values = np.empty((result_unique_labels.size, ncols))
    except IndexError:
        result_reduced_values = np.empty((result_unique_labels.size))

    for i, label in enumerate(result_unique_labels):
        result_reduced_values[i] = numpy_callable(data[data[:,0] == label,1:],axis=0)

    return result_unique_labels, result_reduced_values

# ------------------------------ MAIN FUNCTION ------------------------------ #

# -- Parallel efficiency on GP3.
times_gp3 = np.loadtxt("../../bin_tmp/ParallelEfficiency/times_gp3.dat")
nprocs, times_reduced_gp3 = ReduceDuplicateEntries(times_gp3)

parallel_efficiency_gp3 = np.zeros_like(times_reduced_gp3)
for i in range(times_reduced_gp3.shape[0]):
    parallel_efficiency_gp3[i] = times_reduced_gp3[0]/(times_reduced_gp3[i]*nprocs[i])

# -- We prepare the plot area.
bar_width = 1e-1                          # Width of the bars in the bar plot.

fig = plt.figure(figsize=(4,2.5))
ax = fig.add_subplot(111)
vphys.adjust_spines(ax, ['left', 'bottom'], 2)
#ax.spines['bottom'].set_position(('outward',0.0))

ax.set_xlim(0.75,1.5*np.amax(nprocs))
ax.set_ylim(0,1.05)

ax.set_xscale('log')
ax.xaxis.set_major_locator(ticker.LogLocator(base=2))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:1.0f}"))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:1.1f}"))

ax.set_xticks(nprocs)
ax.set_xlabel(r"Number of processors")
ax.set_ylabel(r"Parallel Efficiency on GP3", rotation='horizontal', ha='left')
ax.yaxis.set_label_coords(-0.1,1.05)

ax.grid(True,color='gray', linestyle="-", lw=0.3, zorder=0)

# -- Remove minor tick marks.
for toc in ax.xaxis.get_minor_ticks():
    toc.tick1On = toc.tick2On = False

width = 1e-1
xdelta = 0.0
vphys.BarPlotWithLogAxes(ax,nprocs,parallel_efficiency_gp3,width, fc='#f0b64d', ec='k', lw=1, zorder=10)

plt.savefig("ParallelEfficiency-gp3.pdf", bbox_inches='tight')
plt.close()

# ------------------------ Parallel Efficiency on mp2 ------------------------ #
nprocs, times_mp2_io = np.loadtxt("../../bin_tmp/ParallelEfficiency/times_mp2_io.dat",unpack=True)
print(nprocs)
nprocs, times_mp2_noio = np.loadtxt("../../bin_tmp/ParallelEfficiency/times_mp2_noio.dat",unpack=True)

parallel_efficiency_mp2_io   = np.empty((nprocs.size))
parallel_efficiency_mp2_noio = np.empty((nprocs.size))

for i in range(times_mp2_io.size):
    parallel_efficiency_mp2_io[i]   = times_mp2_io[0]/(times_mp2_io[i]*nprocs[i])
    parallel_efficiency_mp2_noio[i] = times_mp2_noio[0]/(times_mp2_noio[i]*nprocs[i])

fig = plt.figure(figsize=(4,2.5))
ax  = fig.add_subplot(111)
vphys.adjust_spines(ax, ['left', 'bottom'], 2)

ax.set_xlim(0.75,1.5*np.amax(nprocs))
ax.set_ylim(0,1.05)

ax.set_xscale('log')
ax.xaxis.set_major_locator(ticker.LogLocator(base=2))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:1.0f}"))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:1.1f}"))

ax.set_xticks(nprocs)
ax.set_xlabel(r"Number of processors")
ax.set_ylabel(r"Parallel Efficiency on mp2", rotation='horizontal', ha='left')
ax.yaxis.set_label_coords(-0.1,1.05)

ax.grid(True,color='gray', linestyle="-", lw=0.3, zorder=0)

# -- Remove minor tick marks.
for toc in ax.xaxis.get_minor_ticks():
    toc.tick1On = toc.tick2On = False

width = 0.5e-1
xdelta = width

without_io = vphys.BarPlotWithLogAxes(ax,nprocs,parallel_efficiency_mp2_noio,width,xdelta, fc='#ea5f94', ec='k', lw=1, zorder=10, clip_on=False, label = "Without I/O")
with_io    = vphys.BarPlotWithLogAxes(ax,nprocs,parallel_efficiency_mp2_io,width,-xdelta, fc='#f0b64d', ec='k', lw=1, zorder=10 , clip_on=False, label = "With I/O")
plt.legend(handles=[without_io,with_io], ncol=2,fontsize=7,bbox_to_anchor=(0.91, 1.00),bbox_transform=plt.gcf().transFigure)

plt.savefig("ParallelEfficiency-mp2.pdf", bbox_inches='tight')
plt.close()