# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created:      Mar. 2nd, 2016                                                #
# Modified:     Mar. 2nd, 2016                                                #
# Description:  We draw the limits of the external, classical field approx.   #
#               in SF-QED.                                                    #
# Dependencies: - NumPy                                                       #
#               - SciPy
#               - Matplotlib                                                  #
# --------------------------------------------------------------------------- #

# --------------------------- Modules Importation --------------------------- #
import matplotlib as mpl
mpl.use('pgf')
import numpy as np
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import scipy.constants as cst
from mpl_toolkits.axes_grid1 import make_axes_locatable

import imp
# -- Load our custom modules.
vphys           = imp.load_source('vphys', "../../python-tools/vphys.py")
import vphys


# ------------------------------ Configuration ------------------------------ #
#-- We reset the LaTeX parameters to enable XeLaTeX.
mpl.rcParams.update(vphys.default_pgf_configuration())

# -------------------------------- Variables -------------------------------- #

omega     = np.logspace(-2,7,100) # Angular frequency in electronvolts
intensity = np.logspace(0,32,100) # Intensity in W/cm^2.
O, I = np.meshgrid(omega,intensity)

# -- xi = eE_0/(omega*m_e*c) = np.sqrt(2*I/(epsilon_0*c))/(omega[ev]/hbar*m_e*c).
# -- The 1e4 factor is to convert from W/m^2 to W/cm^2.
xi    = np.sqrt(2e4*I/(cst.c*cst.epsilon_0))/(O/cst.hbar*cst.m_e*cst.c)

# -- Prepare the plot.
fig1 = plt.figure(figsize=(4.666,2.6666))
ax1  = fig1.add_subplot(111)

# -- Draw the "classical" limit and the pair-production
# V. B. Berestetskii, E. M. Lifshitz, and L. P. Pitaezskii, Quantum Electrodynamics, Second  Edition (Pergamon Press, 1982), §44.
ax1.plot(omega, 2.519e3*omega**4, 'k-', lw=2)
ax1.text(0.55,0.55, r"$I \sim 2.5\times10^3\,\si{\watt\per\cm\squared}\times\omega^4$", horizontalalignment='center',color='k', transform=ax1.transAxes, rotation=35)

ax1.axhline(2.25e29, ls='-', lw=2, color='k')
ax1.text(0.01,0.94, r"Schwinger pair production $(I=I_S)$", horizontalalignment='left', color='k', transform=ax1.transAxes,fontsize=8)

# -- Draw \xi and the limit \xi = 1
im = ax1.pcolormesh(O,I,xi, norm=LogNorm(vmin=xi.min(), vmax=xi.max()), cmap=vphys.morgenstemning_cmap, rasterized=True)
ax1.plot(omega, 1e6*cst.hbar/cst.e*np.sqrt(2e4*intensity/(cst.c*cst.epsilon_0))/(cst.m_e*cst.c), 'k--', lw=2)
ax1.text(0.4,0.75, r"$\xi=1$", horizontalalignment='center', color='k', rotation=18, transform=ax1.transAxes)

divider = make_axes_locatable(ax1)
cax     = divider.append_axes("right", size='3%', pad=0.1)
cbar    = plt.colorbar(im, cax=cax)

# -- Other information
ax1.axvline(511e3, color='k')        # Electron mass
ax1.text(0.88,0.5, r"$\omega=m_e$", horizontalalignment='center', rotation=270, transform=ax1.transAxes, color='k')

# -- Axes
ax1.set_xlabel(r"$\omega$ [\si{\eV}]")
ax1.set_ylabel(r"$I$ [\si{\watt\per\cm\squared}]")
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim((1e0,1e32))


plt.savefig("ClassicalLimit.pdf", bbox_inches='tight', dpi=500)
