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
import scipy.integrate as integrate

# -- OS and other stuff.
import time
import argparse
import imp

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
def GaussIntegrals(analysis_obj, field_type, r_index, frequency_index):
  """
  Evaluates the Gaussian integral of Maxwell's equations, either for the
  magnetic or electric field, for a given frequency, and for a given value of r.
  """
  if field_type == "electric":
    radial_field = "Er"
    long_field   = "Ez"
  elif field_type == "magnetic":
    radial_field = "Br"
    long_field  = "Bz"

  # -- Compute the integrands in cylindrical coordinates.
  Fr = analysis_obj.GetFrequencyComponent(radial_field, frequency_index)
  Fz = analysis_obj.GetFrequencyComponent(long_field,   frequency_index)

  cylinderIntegrand = np.zeros_like(Fr)
  circleIntegrand   = np.zeros_like(Fr)

  # -- In cylindrical coordinates, the integrand must be multiplied by r.
  for i in range(cylinderIntegrand.shape[0]):
    cylinderIntegrand[i,:,:] =  analysis_obj.coord_r[i]*Fr[i,:,:]
    circleIntegrand[i,:,:]   = -analysis_obj.coord_r[i]*Fz[i,:,:]

  cylinderContribution  =  integrate.simps(integrate.simps(cylinderIntegrand[r_index,:,:],
                                           x=analysis_obj.coord_z[:]),
                                           x=analysis_obj.coord_theta[:])

  negCircleContribution = -integrate.simps(integrate.simps(circleIntegrand[0:r_index-1,:,0],
                                           x=analysis_obj.coord_theta[:]),
                                           x=analysis_obj.coord_r[0:r_index-1])

  posCircleContribution = integrate.simps(integrate.simps(circleIntegrand[0:r_index-1,:,-1],
                                           x=analysis_obj.coord_theta[:]),
                                           x=analysis_obj.coord_r[0:r_index-1])

  print(np.amax(cylinderIntegrand[r_index,:,:]))
  return (cylinderContribution+negCircleContribution+posCircleContribution)/np.abs(np.amax(cylinderIntegrand[r_index,:,:]))

def SurfaceLineIntegrals(analysis_obj, field_type, r_index, z_index, frequency_index):
  """
  Evaluates the flux thorugh a circular surface, and the current over the loop.
  """
  if field_type == "electric":
    factor = 1
    theta_field = "Eth"
    long_field   = "Bz"
  elif field_type == "magnetic":
    factor = -1
    theta_field = "Bth"
    long_field  = "Ez"

  Fth = analysis_obj.GetFrequencyComponent(theta_field, frequency_index)
  Fz  = analysis_obj.GetFrequencyComponent(long_field, frequency_index)

  lineIntegrand   = np.zeros_like(Fz)
  circleIntegrand = np.zeros_like(Fz)

  # -- In cylindrical coordinates, the integrand must be multiplied by r.
  for i in range(lineIntegrand.shape[0]):
    lineIntegrand[i,:,:]   = analysis_obj.coord_r[i]*Fth[i,:,:]
    circleIntegrand[i,:,:] = analysis_obj.coord_r[i]*Fz[i,:,:]

  lineContribution   = integrate.simps(lineIntegrand[r_index,:,z_index],
                                       x=analysis_obj.coord_theta[:])

  circleContribution = integrate.simps(integrate.simps(circleIntegrand[0:r_index-1,:,z_index],
                                       x=analysis_obj.coord_theta[:]),
                                       x=analysis_obj.coord_r[0:r_index-1])

  plt.plot(lineIntegrand[r_index,:,z_index])
  plt.savefig("lineIntegrand.pdf")
  print(1j*analysis_obj.omega[frequency_index]*analysis_obj.UNIT_TIME*circleContribution)
  return lineContribution-1j*analysis_obj.omega[frequency_index]*analysis_obj.UNIT_TIME*factor*circleContribution

# ------------------------------ MAIN FUNCTION ------------------------------ #
# -- Open the file for analysis.
analysis_object = analstrat.Analysis3D(freq_field="../../bin/MaxwellVerification/Field_reflected.hdf5")

print("The surface integral for the electric case yields: {}".format(GaussIntegrals(analysis_object,
                                                                                    "electric",
                                                                                    analysis_object.size_r//2,
                                                                                    analysis_object.size_freq//2)))

print("The surface integral for the magnetic case yields: {}".format(GaussIntegrals(analysis_object,
                                                                                    "magnetic",
                                                                                    analysis_object.size_r//2,
                                                                                    analysis_object.size_freq//2)))

print("The line integral for the electric case yields: {}".format(SurfaceLineIntegrals(analysis_object,
                                                                                       "electric",
                                                                                       analysis_object.size_r//2,
                                                                                       analysis_object.size_z//2,
                                                                                       analysis_object.size_freq//2)))

print("The line integral for the magnetic case yields: {}".format(SurfaceLineIntegrals(analysis_object,
                                                                                       "magnetic",
                                                                                       analysis_object.size_r//2,
                                                                                       analysis_object.size_z//2,
                                                                                       analysis_object.size_freq//2)))
