# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created:      Dec. 10th, 2015                                               #
# Description:  We analyse the convergence of the StrattoCalculator.          #
# Dependencies: - NumPy                                                       #
#               - SciPy                                                       #
#               - H5Py                                                        #
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
import h5py
from scipy import interpolate
import scipy.integrate as integrate

# -- OS and other stuff.
import os
import time
import argparse
import imp
import glob
import configparser
import re

# -- Load our custom modules.
vphys           = imp.load_source('vphys', "../../python-tools/vphys.py")
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

# --------------------------- Function Definition --------------------------- #
# -- We define a function that formats the folder name string.             -- #
# --------------------------------------------------------------------------- #

def ListSimulationDirectories(bin_dir):
  """
  We count the number of directory that end in \d{5}.BQ. This gives us the
  number of simulation that we ran, and also their names.
  """
  dirList = [f for f in os.listdir(bin_dir) if re.search(r'(.*\d{5}.BQ)', f)]

  sortedList = sorted(dirList, key=str.lower)

  for i in range(len(sortedList)):
    sortedList[i] += "/{:05g}.BQ/".format(i+1)

  return sortedList

def ComputeTotalEnergyDensityTemporalRadial(file_id, r_dat, z_dat, timeIdx):
  """
  We compute the total energy of the system by integrating over the volume of the
  focus. Since we suppose that all the energy is contained in this volume, the final
  value should not depend on the timeIdx, as long as the whole field is present there.
  """

  # -- We prepare the arrays.
  r_dat_SI   = r_dat[:]*analstrat.UNIT_LENGTH
  z_dat_SI   = z_dat[:]*analstrat.UNIT_LENGTH
  integrand  = np.zeros((r_dat_SI.size,z_dat_SI.size))

  Er         = file_id['/field/Er-{}'.format(timeIdx)]
  Ez         = file_id['/field/Ez-{}'.format(timeIdx)]
  Bth        = file_id['/field/Bth-{}'.format(timeIdx)]

  integrand   = 0.5*(Er[:]**2+Ez[:]**2+Bth[:]**2)
  for i in range(integrand.shape[0]):
    integrand[i,:] *= r_dat[i]

  return 2.0*np.pi*integrate.simps(integrate.simps(integrand,x=z_dat[:]),x=r_dat[:])*analstrat.UNIT_MASS*analstrat.SPEED_OF_LIGHT**2

def ComputeTotalEnergyDensityTemporalLinear(file_id, r_dat, theta_dat, z_dat, timeIdx):
  r_dat_SI     = r_dat[:]*UNIT_LENGTH
  theta_dat_SI = theta_dat[:]
  z_dat_SI     = z_dat[:]*UNIT_LENGTH
  size         = np.ceil(z_dat_SI.size/2)

  Er           = file_id['/field/Er-{}'.format(timeIdx)]
  Eth          = file_id['/field/Eth-{}'.format(timeIdx)]
  Ez           = file_id['/field/Ez-{}'.format(timeIdx)]
  Br           = file_id['/field/Br-{}'.format(timeIdx)]
  Bth          = file_id['/field/Bth-{}'.format(timeIdx)]
  Bz           = file_id['/field/Bz-{}'.format(timeIdx)]

  integrand = 0.5*(Er[:]**2+Eth[:]**2+Ez[:]**2+Br[:]**2+Bth[:]**2+Bz[:]**2)

  for i in range(integrand.shape[0]):
    integrand[i,:,:] *= r_dat[i]

  return integrate.simps(integrate.simps(integrate.simps(integrand, x=z_dat[:]), x=theta_dat[:]), x=r_dat[:])*analstrat.UNIT_MASS*analstrat.SPEED_OF_LIGHT**2

# ------------------------------ MAIN FUNCTION ------------------------------ #

# -- We analyze the radial beam simulations.
# -- We will first load the reference simulation in memory, then compute the
# -- absolute difference between the reference simulation and other simulation
# -- as a function of the average cell area. We then plot the convergence rate
# -- to check that it matches the integrator's.
bin_folder = "../../bin/Convergence/Stratto/Radial/"
configFile = bin_folder+"stratto-convergence.ini"
dirList = ListSimulationDirectories(bin_folder)

# -- Load the ref. simulation in memory.
refFolderName = bin_folder+dirList[-1]
hdf5File = h5py.File(refFolderName+"/Field_reflected_time.hdf5", 'r')
hdf5Freq = h5py.File(refFolderName+"/Field_reflected.hdf5", 'r')

size_freq = hdf5Freq['/spectrum'].attrs.get("num_spectral_components")[0]
size_time = len(hdf5File['/time'])

r_dat = hdf5Freq['/coordinates/r']
z_dat = hdf5Freq['/coordinates/z']

energy = np.zeros((size_time))
for i in range(size_time):
  energy[i] = ComputeTotalEnergyDensityTemporalRadial(hdf5File,r_dat,z_dat,i)
refIndex  = np.argmax(energy)
refEnergy = energy[refIndex]

refIndexFreq = size_freq//2
refEr     = hdf5Freq['/field/Er-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Er-{}/phase'.format(refIndexFreq)][:])
refEz     = hdf5Freq['/field/Ez-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Ez-{}/phase'.format(refIndexFreq)][:])
refBth    = hdf5Freq['/field/Bth-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Bth-{}/phase'.format(refIndexFreq)][:])

hdf5File.close()
hdf5Freq.close()

# -- We now compare the reference results to the others.
# -- We first define the quantities that we wish to plot.
size_loop = len(dirList)-1
area       = np.zeros((size_loop))
area_plot  = np.zeros((size_loop))
energyInt  = np.zeros((size_loop))

minDiffEr  = np.zeros((size_loop))
avgDiffEr  = np.zeros((size_loop))
maxDiffEr  = np.zeros((size_loop))

minDiffEz  = np.zeros((size_loop))
avgDiffEz  = np.zeros((size_loop))
maxDiffEz  = np.zeros((size_loop))

minDiffBth = np.zeros((size_loop))
avgDiffBth = np.zeros((size_loop))
maxDiffBth = np.zeros((size_loop))

for idx in range(size_loop):

  # -- We compute thesize average area of a cell on the parabola as a function
  # -- of the number of intervals.
  # Open the config file.
  config       = configparser.ConfigParser(inline_comment_prefixes=";")
  config.read(bin_folder+"/"+dirList[idx]+"stratto-convergence.ini")

  # Read the number of intervals.
  intervals_r   = int(config['Parabola']['intervals_r'])
  num_points_r  = int(config['Parabola']['num_points_r'])
  intervals_th  = int(config['Parabola']['intervals_th'])
  num_points_th = int(config['Parabola']['num_points_th'])

  # This assumes a Gauss-Legendre mesh.
  N_r        = intervals_r*num_points_r
  N_th       = intervals_th*num_points_th
  deltaTheta = math.radians(float(config['Parabola']['th_max'])-float(config['Parabola']['th_min']))/N_th
  deltaR     = (float(config['Parabola']['r_max'])-float(config['Parabola']['r_min']))/N_r
  area[idx]  = deltaTheta*deltaR**2*(N_r+1)**2/(2*N_r)

  # -- We open the proper files.
  folderName = bin_folder+dirList[idx]
  hdf5File   = h5py.File(folderName+"/Field_reflected_time.hdf5", 'r')
  hdf5Freq   = h5py.File(folderName+"/Field_reflected.hdf5", 'r')

  # -- Open the mesh in the focal area.
  r_dat          = hdf5File['/coordinates/r']
  z_dat          = hdf5File['/coordinates/z']
  area_plot[idx] = area[idx]/((hdf5Freq['/spectrum/wavelength (m)'][refIndexFreq])**2)

  # -- We compute the energy at the reference index.
  energyInt[idx] = ComputeTotalEnergyDensityTemporalRadial(hdf5File,r_dat,z_dat,refIndex)
  print(energyInt[idx], energyInt[idx]-refEnergy, refEnergy)

  # -- We compute the difference of the frequency components, but keep only the
  # -- minimum, average and maximum.
  absDiffEr  = np.abs(refEr-hdf5Freq['/field/Er-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Er-{}/phase'.format(refIndexFreq)][:]))
  absDiffEz  = np.abs(refEz-hdf5Freq['/field/Ez-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Ez-{}/phase'.format(refIndexFreq)][:]))
  absDiffBth = np.abs(refBth-hdf5Freq['/field/Bth-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Bth-{}/phase'.format(refIndexFreq)][:]))

  minDiffEr[idx]  = np.amin(absDiffEr)
  avgDiffEr[idx]  = np.mean(absDiffEr)
  maxDiffEr[idx]  = np.amax(absDiffEr)

  minDiffEz[idx]  = np.amin(absDiffEz)
  avgDiffEz[idx]  = np.mean(absDiffEz)
  maxDiffEz[idx]  = np.amax(absDiffEz)

  minDiffBth[idx] = np.amin(absDiffBth)
  avgDiffBth[idx] = np.mean(absDiffBth)
  maxDiffBth[idx] = np.amax(absDiffBth)

  # -- Some bookkeeping for the loop.
  hdf5File.close()
  hdf5Freq.close()

# ------------------------- Plotting the Convergence ------------------------ #

# -- We check the convergence order of the components.
minIndexConv = 3
maxIndexConv = -1
convXFit      = np.log(area[minIndexConv:maxIndexConv])
ErConvYFit    = np.log(avgDiffEr[minIndexConv:maxIndexConv])
EzConvYFit    = np.log(avgDiffEz[minIndexConv:maxIndexConv])
BthConvYFit   = np.log(avgDiffBth[minIndexConv:maxIndexConv])

# Polynomial fits.
ErConvPoly    = np.polyfit(convXFit, ErConvYFit, 1)
EzConvPoly    = np.polyfit(convXFit, EzConvYFit, 1)
BthConvPoly   = np.polyfit(convXFit, BthConvYFit, 1)

print("Order of convergence for Er is {}".format(ErConvPoly[0]))
print("Order of convergence for Ez is {}".format(EzConvPoly[0]))
print("Order of convergence for Bth is {}".format(BthConvPoly[0]))

# Fitted values.
ErConvFitted  = np.poly1d(ErConvPoly)(convXFit)
EzConvFitted  = np.poly1d(EzConvPoly)(convXFit)
BthConvFitted = np.poly1d(BthConvPoly)(convXFit)

# -- We check the convergence of the components.
figCompConv, ((axErConv, axEzConv), (axBthConv, axConv))  = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(3,2.5))
figCompConv.subplots_adjust(hspace=0,wspace=0)

#axErConv = figCompConv.add_subplot(141)
figCompConv.suptitle("Average cell length [$/\lambda$]", y=0)

# -- Er.
#axErConv    = figCompConv.add_subplot(141)
#axErConv.plot(np.sqrt(area), minDiffEr)
#axErConv.plot(np.sqrt(area), avgDiffEr)
axErConv.plot(np.sqrt(area_plot)[minIndexConv:maxIndexConv], maxDiffEr[minIndexConv:maxIndexConv],       'b^', zorder=3)
axErConv.plot(np.sqrt(area_plot)[maxIndexConv:],  maxDiffEr[maxIndexConv:],        'b^', zorder=3, markevery=5)
axErConv.plot(np.sqrt(area_plot)[minIndexConv:maxIndexConv], np.exp(ErConvFitted), 'k',  zorder=3)

#axErConv.set_title(r'$E_r$')
axErConv.set_xscale('log')
axErConv.set_yscale('log')
axErConv.set_xlim((10**(1.5),10**(4.5)))
axErConv.xaxis.set_ticks((1e2,1e3,1e4))
axErConv.invert_xaxis()

axErConv.set_ylabel("Absolute\n error", rotation='horizontal', va='center', ha='left')
axErConv.yaxis.set_label_coords(-0.2,1.15)
axErConv.text(0.03,0.83, "(a)", transform=axErConv.transAxes)#, backgroundcolor='white')
axErConv.text(0.85,0.83, "$E_r$", transform=axErConv.transAxes)#, backgroundcolor='white')

axErConv.grid(True,color='gray', zorder=0)

# -- Ez.
# axEzConv   = figCompConv.add_subplot(142, sharey=axErConv)
#axEzConv.plot(np.sqrt(area), minDiffEz)
#axEzConv.plot(np.sqrt(area), avgDiffEz)
axEzConv.plot(np.sqrt(area_plot)[minIndexConv:maxIndexConv], maxDiffEz[minIndexConv:maxIndexConv],       'b^', zorder=3)
axEzConv.plot(np.sqrt(area_plot)[maxIndexConv:],  maxDiffEz[maxIndexConv:],        'b^', zorder=3, markevery=5)
axEzConv.plot(np.sqrt(area_plot)[minIndexConv:maxIndexConv], np.exp(EzConvFitted), 'k',  zorder=3)

#axEzConv.set_title(r'$E_z$')
axEzConv.set_xscale('log')
axEzConv.set_xlim((10**(1.5),10**(4.5)))
axEzConv.xaxis.set_ticks((1e2,1e3,1e4))
axEzConv.invert_xaxis()

axEzConv.text(0.03,0.83, "(b)", transform=axEzConv.transAxes)#, backgroundcolor='white')
axEzConv.text(0.85,0.83, "$E_z$", transform=axEzConv.transAxes)#, backgroundcolor='white')

axEzConv.grid(True, color='gray', zorder=0)

# -- Bth
#axBthConv  = figCompConv.add_subplot(143, sharey=axErConv)
#axBthConv.plot(np.sqrt(area), minDiffBth)
#axBthConv.plot(np.sqrt(area), avgDiffBth)
axBthConv.plot(np.sqrt(area_plot)[minIndexConv:maxIndexConv], maxDiffBth[minIndexConv:maxIndexConv],       'b^', zorder=3)
axBthConv.plot(np.sqrt(area_plot)[maxIndexConv:],  maxDiffBth[maxIndexConv:],        'b^', zorder=3, markevery=5)
axBthConv.plot(np.sqrt(area_plot)[minIndexConv:maxIndexConv], np.exp(BthConvFitted), 'k', zorder=3)

#axBthConv.set_title(r'$B_\theta$')
axBthConv.set_xscale('log')
axBthConv.set_xlim((10**(1.5),10**(4.5)))
axBthConv.xaxis.set_ticks((1e2,1e3,1e4))
axBthConv.invert_xaxis()

axBthConv.text(0.03,0.83, "(c)", transform=axBthConv.transAxes)#, backgroundcolor='white')
axBthConv.text(0.85,0.83, r"$B_\theta$", transform=axBthConv.transAxes)#, backgroundcolor='white')

axBthConv.grid(True, color='gray', zorder=0)

#plt.savefig("ComponentsConvergence.pdf", bbox_inches='tight')

# -- We check the convergence of the energy.
# -- We fit a line on the log-log graph.
xfit  = np.log(np.sqrt(area[minIndexConv:maxIndexConv]))
yfit  = np.log(np.abs(energyInt-refEnergy)[minIndexConv:maxIndexConv])
polyno= np.polyfit(xfit,yfit,1)
pfit  = np.poly1d(polyno)
fitplt= pfit(xfit)

print("Order of convergence of the energy is {}".format(polyno[0]))

#figConv  = plt.figure(figsize=(3,3))
#axConv   = figCompConv.add_subplot(144, sharey=axErConv)
axConv.plot(np.sqrt(area_plot)[minIndexConv:maxIndexConv],np.abs(energyInt-refEnergy)[minIndexConv:maxIndexConv], 'b^', zorder=3)
axConv.plot(np.sqrt(area_plot)[maxIndexConv:], np.abs(energyInt-refEnergy)[maxIndexConv:],  'b^', zorder=3, markevery=5)
axConv.plot(np.sqrt(area_plot)[minIndexConv:maxIndexConv],np.exp(fitplt),                    'k', zorder=3)

#axConv.set_title("Energy")
axConv.set_xscale('log')
axConv.set_xlim((10**(1.5),10**(4.5)))
axConv.xaxis.set_ticks((1e2,1e3,1e4))
axConv.invert_xaxis()
axConv.text(0.03,0.83, "(d)", transform=axConv.transAxes)#,backgroundcolor='white')
axConv.text(0.95,0.85, "Energy", transform=axConv.transAxes, ha='right')


#plt.savefig("convergence.pdf", bbox_inches='tight')
plt.savefig("ConvergenceAll.pdf", bbox_inches='tight')

# --------------------------------------------------------------------------- #
# We analyze the linear beam simulations.                                     #
# We will first load the reference simulation in memory, then compute the     #
# absolute difference between the reference simulation and other simulation   #
# as a function of the average cell area. We then plot the convergence rate   #
# to check that it matches the integrator's.                                  #
# --------------------------------------------------------------------------- #
bin_folder = "../../bin/Convergence/Stratto/Linear/"
configFile = bin_folder+"stratto-convergence.ini"
dirList = ListSimulationDirectories(bin_folder)

# -- Load the ref. simulation in memory.
refFolderName = bin_folder+dirList[-1]
hdf5File = h5py.File(refFolderName+"/Field_reflected_time.hdf5", 'r')
hdf5Freq = h5py.File(refFolderName+"/Field_reflected.hdf5", 'r')
refIndexFreq = size_freq//2
refEr     = hdf5Freq['/field/Er-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Er-{}/phase'.format(refIndexFreq)][:])
refEth    = hdf5Freq['/field/Eth-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Eth-{}/phase'.format(refIndexFreq)][:])
refEz     = hdf5Freq['/field/Ez-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Ez-{}/phase'.format(refIndexFreq)][:])
refBr     = hdf5Freq['/field/Br-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Br-{}/phase'.format(refIndexFreq)][:])
refBth    = hdf5Freq['/field/Bth-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Bth-{}/phase'.format(refIndexFreq)][:])
refBz     = hdf5Freq['/field/Bz-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['field/Bz-{}/phase'.format(refIndexFreq)][:])

hdf5File.close()
hdf5Freq.close()

# -- We now compare the reference results to the others.
# -- We first define the quantities that we wish to plot.
size_loop  = len(dirList)-1
area       = np.zeros((size_loop))
energyInt  = np.zeros((size_loop))

minDiffEr  = np.zeros((size_loop))
avgDiffEr  = np.zeros((size_loop))
maxDiffEr  = np.zeros((size_loop))

minDiffEth = np.zeros((size_loop))
avgDiffEth = np.zeros((size_loop))
maxDiffEth = np.zeros((size_loop))

minDiffEz  = np.zeros((size_loop))
avgDiffEz  = np.zeros((size_loop))
maxDiffEz  = np.zeros((size_loop))

minDiffBr  = np.zeros((size_loop))
avgDiffBr  = np.zeros((size_loop))
maxDiffBr  = np.zeros((size_loop))

minDiffBth = np.zeros((size_loop))
avgDiffBth = np.zeros((size_loop))
maxDiffBth = np.zeros((size_loop))

minDiffBz  = np.zeros((size_loop))
avgDiffBz  = np.zeros((size_loop))
maxDiffBz  = np.zeros((size_loop))

for idx in range(size_loop):
  # -- We compute the size average area of a cell on the parabola as a function
  # -- of the number of intervals.
  # Open the config file.
  config       = configparser.ConfigParser(inline_comment_prefixes=";")
  config.read(bin_folder+"/"+dirList[idx]+"stratto-convergence.ini")

  # Read the number of intervals.
  intervals_r   = int(config['Parabola']['intervals_r'])
  num_points_r  = int(config['Parabola']['num_points_r'])
  intervals_th  = int(config['Parabola']['intervals_th'])
  num_points_th = int(config['Parabola']['num_points_th'])

  # This assumes a Gauss-Legendre mesh.
  N_r        = intervals_r*num_points_r
  N_th       = intervals_th*num_points_th
  deltaTheta = math.radians(float(config['Parabola']['th_max'])-float(config['Parabola']['th_min']))/N_th
  deltaR     = (float(config['Parabola']['r_max'])-float(config['Parabola']['r_min']))/N_r
  area[idx]  = deltaTheta*deltaR**2*(N_r+1)**2/(2*N_r)

  # -- We open the proper files.
  folderName = bin_folder+dirList[idx]
  hdf5File   = h5py.File(folderName+"/Field_reflected_time.hdf5", 'r')
  hdf5Freq   = h5py.File(folderName+"/Field_reflected.hdf5", 'r')

  # -- We compute the difference of the frequency components, but keep only the
  # -- minimum, average and maximum.
  absDiffEr  = np.abs(refEr-hdf5Freq['/field/Er-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Er-{}/phase'.format(refIndexFreq)][:]))
  absDiffEth = np.abs(refEth-hdf5Freq['/field/Eth-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Eth-{}/phase'.format(refIndexFreq)][:]))
  absDiffEz  = np.abs(refEz-hdf5Freq['/field/Ez-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Ez-{}/phase'.format(refIndexFreq)][:]))
  absDiffBr  = np.abs(refBr-hdf5Freq['/field/Br-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Br-{}/phase'.format(refIndexFreq)][:]))
  absDiffBth = np.abs(refBth-hdf5Freq['/field/Bth-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Bth-{}/phase'.format(refIndexFreq)][:]))
  absDiffBz  = np.abs(refBz-hdf5Freq['/field/Bz-{}/amplitude'.format(refIndexFreq)][:]*np.exp(1j*hdf5Freq['/field/Bz-{}/phase'.format(refIndexFreq)][:]))

  minDiffEr[idx]  = np.amin(absDiffEr)
  avgDiffEr[idx]  = np.mean(absDiffEr)
  maxDiffEr[idx]  = np.amax(absDiffEr)

  minDiffEth[idx]  = np.amin(absDiffEth)
  avgDiffEth[idx]  = np.mean(absDiffEth)
  maxDiffEth[idx]  = np.amax(absDiffEth)

  minDiffEz[idx]  = np.amin(absDiffEz)
  avgDiffEz[idx]  = np.mean(absDiffEz)
  maxDiffEz[idx]  = np.amax(absDiffEz)

  minDiffBr[idx]  = np.amin(absDiffBr)
  avgDiffBr[idx]  = np.mean(absDiffBr)
  maxDiffBr[idx]  = np.amax(absDiffBr)

  minDiffBth[idx] = np.amin(absDiffBth)
  avgDiffBth[idx] = np.mean(absDiffBth)
  maxDiffBth[idx] = np.amax(absDiffBth)

  minDiffBz[idx]  = np.amin(absDiffBz)
  avgDiffBz[idx]  = np.mean(absDiffBz)
  maxDiffBz[idx]  = np.amax(absDiffBz)

  # -- Some bookkeeping for the loop.
  hdf5File.close()
  hdf5Freq.close()

# ------------------------- Plotting the Convergence ------------------------ #

# -- We check the convergence order of the components.
convXFit      = np.log(np.sqrt(area[3:-2]))
ErConvYFit    = np.log(avgDiffEr[3:-2])
EthConvYFit   = np.log(avgDiffEth[3:-2])
EzConvYFit    = np.log(avgDiffEz[3:-2])
BrConvYFit    = np.log(avgDiffBr[3:-2])
BthConvYFit   = np.log(avgDiffBth[3:-2])
BzConvYFit    = np.log(avgDiffBz[3:-2])

# Polynomial fits.
ErConvPoly    = np.polyfit(convXFit, ErConvYFit, 1)
EthConvPoly   = np.polyfit(convXFit, EthConvYFit, 1)
EzConvPoly    = np.polyfit(convXFit, EzConvYFit, 1)
BrConvPoly    = np.polyfit(convXFit, BrConvYFit, 1)
BthConvPoly   = np.polyfit(convXFit, BthConvYFit, 1)
BzConvPoly    = np.polyfit(convXFit, BzConvYFit, 1)

print("Order of convergence for Er is {}".format(ErConvPoly[0]))
print("Order of convergence for Eth is {}".format(EthConvPoly[0]))
print("Order of convergence for Ez is {}".format(EzConvPoly[0]))
print("Order of convergence for Br is {}".format(BrConvPoly[0]))
print("Order of convergence for Bth is {}".format(BthConvPoly[0]))
print("Order of convergence for Bz is {}".format(BzConvPoly[0]))

# Fitted values.
ErConvFitted  = np.poly1d(ErConvPoly)(convXFit)
EthConvFitted = np.poly1d(EthConvPoly)(convXFit)
EzConfFitted  = np.poly1d(EzConvPoly)(convXFit)
BrConvFitted  = np.poly1d(BrConvPoly)(convXFit)
BthConvFitted = np.poly1d(BthConvPoly)(convXFit)
BzConvFitted  = np.poly1d(BzConvFitted)(convXFit)

# -- We check the convergence of the components.
figCompConv, ((axErConv, axBrConv),(axEthConv, axBthConv),(axEzConv,  axBzConv))  = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(8,2))
plt.xlim((1e-10,1e1))

# -- Er.
#axErConv    = figCompConv.add_subplot(141)
#axErConv.plot(np.sqrt(area), minDiffEr)
axErConv.plot(np.sqrt(area), avgDiffEr)
axErConv.plot(np.sqrt(area), maxDiffEr)

axErConv.invert_xaxis()
axErConv.set_xscale('log')
axErConv.set_yscale('log')
axErConv.set_ylabel(r"Absolute error")

# -- Eth.
#axEthConv    = figCompConv.add_subplot(141)
#axEthConv.plot(np.sqrt(area), minDiffEr)
axEthConv.plot(np.sqrt(area), avgDiffEth)
axEthConv.plot(np.sqrt(area), maxDiffEth)

axEthConv.invert_xaxis()
axEthConv.set_xscale('log')
axEthConv.set_yscale('log')
axEthConv.set_ylabel(r"Absolute error")

# -- Ez.
#axEzConv   = figCompConv.add_subplot(142)
#axEzConv.plot(np.sqrt(area), minDiffEz)
axEzConv.plot(np.sqrt(area), avgDiffEz)
axEzConv.plot(np.sqrt(area), maxDiffEz)

axEzConv.invert_xaxis()
axEzConv.set_xscale('log')
axEzConv.set_yscale('log')
#axEzConv.set_ylabel(r"Absolute error")

# -- Br
#axBrConv  = figCompConv.add_subplot(143)
#axBrConv.plot(np.sqrt(area), minDiffBth)
axBrConv.plot(np.sqrt(area), avgDiffBr)
axBrConv.plot(np.sqrt(area), maxDiffBr)

axBrConv.invert_xaxis()
axBrConv.set_xscale('log')
axBrConv.set_yscale('log')
#axBrConv.set_ylabel(r"Absolute error")
axBrConv.set_xlabel(r"Average cell length")

# -- Bth
#axBthConv  = figCompConv.add_subplot(143)
#plt.plot(np.sqrt(area), minDiffBth)
plt.plot(np.sqrt(area), avgDiffBth)
plt.plot(np.sqrt(area), maxDiffBth)

axBthConv.invert_xaxis()
axBthConv.set_xscale('log')
axBthConv.set_yscale('log')
#axBthConv.set_ylabel(r"Absolute error")
axBthConv.set_xlabel(r"Average cell length")

# -- Bz
#axBzConv  = figCompConv.add_subplot(143)
#axBzConv.plot(np.sqrt(area), minDiffBth)
axBzConv.plot(np.sqrt(area), avgDiffBz)
axBzConv.plot(np.sqrt(area), maxDiffBz)

axBzConv.invert_xaxis()
axBzConv.set_xscale('log')
axBzConv.set_yscale('log')
#axBzConv.set_ylabel(r"Absolute error")
axBzConv.set_xlabel(r"Average cell length")


#plt.savefig("ComponentsConvergence.pdf", bbox_inches='tight')

plt.savefig("ConvergenceAll-linear.pdf", bbox_inches='tight')
