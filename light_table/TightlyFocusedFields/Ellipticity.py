# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created:      Apr. 12th, 2018                                               #
# Description:  We plot the Stratton-Chu fields and determine some of their   #
#               more interesting properties.
# Dependencies: - NumPy                                                       #
#               - SciPy                                                       #
#               - Matplotlib                                                  #
# --------------------------------------------------------------------------- #

# --------------------------- Modules Importation --------------------------- #
# -- Plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('pgf')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.collections import PolyCollection


# -- Analysis
import numpy as np
import scipy.constants as cst
import math
import h5py
from scipy import interpolate
import scipy.integrate as integrate
from scipy.optimize import curve_fit

# -- OS and other stuff.
import os
import time
import argparse
import imp
import glob
import configparser
import re
import pickle
import subprocess
import shlex

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
parser.add_argument("--force",
                      action="store_true",
                      help="Forces the recalculation from the original dataset (slow).")
args = parser.parse_args()

# --------------------------- Function Definition --------------------------- #

def DataAlreadyPostProcessed(variable_name_str):
    """
    We check whether the data we wish to extract from our HDF5 file has already
    been extracted. All processed data in saved in the data/ subdirectory, and
    ends in .dat. variable_name_str is the stringified variable name we wish
    to check. Returns True if already extract.
    """
    path = variable_name_str
    return os.path.isfile(path)

def ComputeEllipcity(a,c):
  """
  Computes the ellipticity as sqrt(1-(min(a,c)/max(a,c))**2).
  """
  smajor = a if a>c else c
  sminor = c if a>c else a

  return np.sqrt(1-(sminor/smajor)**2)


# ------------------------------ MAIN FUNCTION ------------------------------ #

# -- Some generic parametres.
bin_folder = "/opt/data/thesis-data/Fields/Stratto/na_vsf_lin_g/"
configFile = "stratto-linear.ini"
dirList = vphys.ListSimulationDirectories(bin_folder)

ratio_analyzed_frequency = 2

# -- Geometric properties
focal_length_f           = np.zeros((len(dirList)))
rmax_f                   = np.zeros((len(dirList)))
alpha_f                  = np.zeros((len(dirList)))

# -- Properties to keep track of as a function of the focal length.
intensity_E              = np.zeros((len(dirList)))
intensity_E_waist_x      = np.zeros((len(dirList)))
intensity_E_waist_y      = np.zeros((len(dirList)))
intensity_E_ellipticity  = np.zeros((len(dirList)))
max_EzEx                 = np.zeros((len(dirList)))

intensity_B              = np.zeros((len(dirList)))
intensity_B_waist_x      = np.zeros((len(dirList)))
intensity_B_waist_y      = np.zeros((len(dirList)))
intensity_B_ellipticity  = np.zeros((len(dirList)))
max_BzBx                 = np.zeros((len(dirList)))

for idx, folder in enumerate(dirList):
    # -- We open the proper files.
    folderName = bin_folder+folder
    dataFolder = folderName+"/data/"
    vphys.mkdir_p(folderName+"/data")

    try:
      analysis_obj = analstrat.Analysis3D(freq_field=folderName+"/Field_reflected.hdf5",time_field=folderName+"/Field_reflected_time.hdf5")
    except:
      intensity_E[idx]              = None
      intensity_E_waist_x[idx]      = None
      intensity_E_waist_y[idx]      = None
      intensity_E_ellipticity[idx]  = None
      max_EzEx[idx]                 = None

      intensity_B[idx]              = None
      intensity_B_waist_x[idx]      = None
      intensity_B_waist_y[idx]      = None
      intensity_B_ellipticity[idx]  = None
      max_BzBx[idx]                 = None

      continue
    freq_idx = analysis_obj.size_freq//ratio_analyzed_frequency
    time_idx = analysis_obj.size_time//ratio_analyzed_frequency

    config = configparser.ConfigParser(inline_comment_prefixes=";")
    config.read(folderName+"/"+configFile)

    focal_length     = float(config['Parabola']['focal_length'])
    rmax             = float(config['Parabola']['r_max'])
    alpha            = 2*focal_length/rmax

    focal_length_f[idx] = focal_length
    rmax_f[idx]         = rmax
    alpha_f[idx]        = alpha

    # ---------------- FREQUENCY DOMAIN ----------------- #
    # -- We plot the focal plane.
    extractedFocalPlaneFreq = DataAlreadyPostProcessed(dataFolder+"ExFocalPlaneFreq.npy") \
                          and DataAlreadyPostProcessed(dataFolder+"EyFocalPlaneFreq.npy") \
                          and DataAlreadyPostProcessed(dataFolder+"EzFocalPlaneFreq.npy") \
                          and DataAlreadyPostProcessed(dataFolder+"BxFocalPlaneFreq.npy") \
                          and DataAlreadyPostProcessed(dataFolder+"ByFocalPlaneFreq.npy") \
                          and DataAlreadyPostProcessed(dataFolder+"BzFocalPlaneFreq.npy") \
                          and not args.force

    if (not extractedFocalPlaneFreq):
        ExFocalPlaneFreq, EyFocalPlaneFreq, EzFocalPlaneFreq, BxFocalPlaneFreq, ByFocalPlaneFreq, BzFocalPlaneFreq \
            = analysis_obj.GetFocalPlaneInFreqCartesian(analysis_obj.size_z//2)
        np.save(dataFolder+"ExFocalPlaneFreq.npy", ExFocalPlaneFreq)
        np.save(dataFolder+"EyFocalPlaneFreq.npy", EyFocalPlaneFreq)
        np.save(dataFolder+"EzFocalPlaneFreq.npy", EzFocalPlaneFreq)
        np.save(dataFolder+"BxFocalPlaneFreq.npy", BxFocalPlaneFreq)
        np.save(dataFolder+"ByFocalPlaneFreq.npy", ByFocalPlaneFreq)
        np.save(dataFolder+"BzFocalPlaneFreq.npy", BzFocalPlaneFreq)
    else:
        ExFocalPlaneFreq = np.load(dataFolder+"ExFocalPlaneFreq.npy")
        EyFocalPlaneFreq = np.load(dataFolder+"EyFocalPlaneFreq.npy")
        EzFocalPlaneFreq = np.load(dataFolder+"EzFocalPlaneFreq.npy")
        BxFocalPlaneFreq = np.load(dataFolder+"BxFocalPlaneFreq.npy")
        ByFocalPlaneFreq = np.load(dataFolder+"ByFocalPlaneFreq.npy")
        BzFocalPlaneFreq = np.load(dataFolder+"BzFocalPlaneFreq.npy")


    analstrat.PlotAllFieldComponentsOnAPlane(analysis_obj.X_meshgrid/1e-6,
                                             analysis_obj.Y_meshgrid/1e-6,
                                             np.transpose(np.abs(ExFocalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(EyFocalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(EzFocalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(BxFocalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(ByFocalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(BzFocalPlaneFreq[:,:,freq_idx]))**2,
                                             folderName+"/FocalPlaneFreq.pdf",
                                             normalization=True)

    # Find the maximum of Ex and Ez, and compute their ratios.
    # We do this in the freq domain to sidestep the pi/2 delay in time.
    max_EzEx[idx] = np.amax(np.abs(ExFocalPlaneFreq[:,:,freq_idx]))/np.amax(np.abs(EzFocalPlaneFreq[:,:,freq_idx]))

    # -- Compute the focal area.
    electric_intensity = np.abs(ExFocalPlaneFreq)**2+np.abs(EyFocalPlaneFreq)**2+np.abs(EzFocalPlaneFreq)**2
    magnetic_intensity = np.abs(BxFocalPlaneFreq)**2+np.abs(ByFocalPlaneFreq)**2+np.abs(BzFocalPlaneFreq)**2

    electric_focal_area = analysis_obj.ComputeFocalArea(electric_intensity[:,:,freq_idx],0.5)/(analysis_obj.wavelength[freq_idx]**2)

    x_cut, y_cut, field_xcut, field_ycut = analysis_obj.PrepareTransverseCuts(analysis_obj.X_meshgrid,
                                                                              analysis_obj.Y_meshgrid,
                                                                              electric_intensity[:,:,freq_idx])

    # Image at focal plane.
    axPlane  = plt.subplot2grid((3,3), (0,1), colspan=2,rowspan=2)
    im         = axPlane.pcolormesh(analysis_obj.X_meshgrid*1e6,analysis_obj.Y_meshgrid*1e6, np.transpose(electric_intensity[:,:,freq_idx]), cmap='viridis', rasterized=True)
    axPlane.set_aspect('equal')
    axPlane.set_xticklabels('', visible=False)
    axPlane.set_yticklabels('', visible=False)

    # x-cut of Ex.
    axXCut     = plt.subplot2grid((3,3), (2,1), colspan=2)
    axXCut.set_xlabel(r"$x$ [$\si{\micro\metre}$]")
    axXCut.plot(x_cut*1e6,field_ycut)
    plt.locator_params(axis='y', nbins=4)

    # y-cut of Ex
    axYCut     = plt.subplot2grid((3,3), (0,0), rowspan=2)
    axYCut.set_ylabel(r"$y$ [\si{\micro\metre}]")
    axYCut.plot(field_ycut,y_cut*1e6)
    plt.locator_params(axis='x', nbins=4)

    # Colobar and other decorations.
    box = axPlane.get_position()

    # create color bar
    axColor = plt.axes([box.x0*1.05 + box.width * 1.05, box.y0, 0.01, box.height])
    plt.colorbar(im, cax = axColor, orientation="vertical")
    plt.savefig(folderName+"/ElectricIntensityFreq.pdf", bbox_inches='tight')
    plt.close()


    # -- We plot the sagittal plane
    extractedSagittalPlaneFreq =    DataAlreadyPostProcessed(dataFolder+"ExSagittalPlaneFreq.npy") \
                                and DataAlreadyPostProcessed(dataFolder+"EySagittalPlaneFreq.npy") \
                                and DataAlreadyPostProcessed(dataFolder+"EzSagittalPlaneFreq.npy") \
                                and DataAlreadyPostProcessed(dataFolder+"BxSagittalPlaneFreq.npy") \
                                and DataAlreadyPostProcessed(dataFolder+"BySagittalPlaneFreq.npy") \
                                and DataAlreadyPostProcessed(dataFolder+"BzSagittalPlaneFreq.npy") \
                                and not args.force

    if (not extractedSagittalPlaneFreq):
        ExSagittalPlaneFreq, EySagittalPlaneFreq, EzSagittalPlaneFreq, \
        BxSagittalPlaneFreq, BySagittalPlaneFreq, BzSagittalPlaneFreq   \
            = analysis_obj.GetSagittalPlaneInFreqCartesian()

        np.save(dataFolder+"ExSagittalPlaneFreq.npy", ExSagittalPlaneFreq)
        np.save(dataFolder+"EySagittalPlaneFreq.npy", EySagittalPlaneFreq)
        np.save(dataFolder+"EzSagittalPlaneFreq.npy", EzSagittalPlaneFreq)
        np.save(dataFolder+"BxSagittalPlaneFreq.npy", BxSagittalPlaneFreq)
        np.save(dataFolder+"BySagittalPlaneFreq.npy", BySagittalPlaneFreq)
        np.save(dataFolder+"BzSagittalPlaneFreq.npy", BzSagittalPlaneFreq)
    else:
        ExSagittalPlaneFreq = np.load(dataFolder+"ExSagittalPlaneFreq.npy")
        EySagittalPlaneFreq = np.load(dataFolder+"EySagittalPlaneFreq.npy")
        EzSagittalPlaneFreq = np.load(dataFolder+"EzSagittalPlaneFreq.npy")
        BxSagittalPlaneFreq = np.load(dataFolder+"BxSagittalPlaneFreq.npy")
        BySagittalPlaneFreq = np.load(dataFolder+"BySagittalPlaneFreq.npy")
        BzSagittalPlaneFreq = np.load(dataFolder+"BzSagittalPlaneFreq.npy")

    analstrat.PlotAllFieldComponentsOnAPlane(analysis_obj.R_axial_meshgrid/1e-6,
                                             analysis_obj.Z_axial_meshgrid/1e-6,
                                             np.transpose(np.abs(ExSagittalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(EySagittalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(EzSagittalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(BxSagittalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(BySagittalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(BzSagittalPlaneFreq[:,:,freq_idx]))**2,
                                             folderName+"/SagittalPlaneFreq.pdf",
                                             normalization=True)

    # -- We plot and analyze the meridional plane.
    extractedMeridionalPlaneFreq =  DataAlreadyPostProcessed(dataFolder+"ExMeridionalPlaneFreq.npy") \
                                and DataAlreadyPostProcessed(dataFolder+"EyMeridionalPlaneFreq.npy") \
                                and DataAlreadyPostProcessed(dataFolder+"EzMeridionalPlaneFreq.npy") \
                                and DataAlreadyPostProcessed(dataFolder+"BxMeridionalPlaneFreq.npy") \
                                and DataAlreadyPostProcessed(dataFolder+"ByMeridionalPlaneFreq.npy") \
                                and DataAlreadyPostProcessed(dataFolder+"BzMeridionalPlaneFreq.npy") \
                                and not args.force

    if (not extractedMeridionalPlaneFreq):
        ExMeridionalPlaneFreq, EyMeridionalPlaneFreq, EzMeridionalPlaneFreq, \
        BxMeridionalPlaneFreq, ByMeridionalPlaneFreq, BzMeridionalPlaneFreq  \
            = analysis_obj.GetMeridionalPlaneInFreqCartesian()

        np.save(dataFolder+"ExMeridionalPlaneFreq.npy", ExMeridionalPlaneFreq)
        np.save(dataFolder+"EyMeridionalPlaneFreq.npy", EyMeridionalPlaneFreq)
        np.save(dataFolder+"EzMeridionalPlaneFreq.npy", EzMeridionalPlaneFreq)
        np.save(dataFolder+"BxMeridionalPlaneFreq.npy", BxMeridionalPlaneFreq)
        np.save(dataFolder+"ByMeridionalPlaneFreq.npy", ByMeridionalPlaneFreq)
        np.save(dataFolder+"BzMeridionalPlaneFreq.npy", BzMeridionalPlaneFreq)
    else:
        ExMeridionalPlaneFreq = np.load(dataFolder+"ExMeridionalPlaneFreq.npy")
        EyMeridionalPlaneFreq = np.load(dataFolder+"EyMeridionalPlaneFreq.npy")
        EzMeridionalPlaneFreq = np.load(dataFolder+"EzMeridionalPlaneFreq.npy")
        BxMeridionalPlaneFreq = np.load(dataFolder+"BxMeridionalPlaneFreq.npy")
        ByMeridionalPlaneFreq = np.load(dataFolder+"ByMeridionalPlaneFreq.npy")
        BzMeridionalPlaneFreq = np.load(dataFolder+"BzMeridionalPlaneFreq.npy")

    analstrat.PlotAllFieldComponentsOnAPlane(analysis_obj.R_axial_meshgrid/1e-6,
                                             analysis_obj.Z_axial_meshgrid/1e-6,
                                             np.transpose(np.abs(ExMeridionalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(EyMeridionalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(EzMeridionalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(BxMeridionalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(ByMeridionalPlaneFreq[:,:,freq_idx]))**2,
                                             np.transpose(np.abs(BzMeridionalPlaneFreq[:,:,freq_idx]))**2,
                                             folderName+"/MeridionalPlaneFreq.pdf",
                                             normalization=True)

    # ------------------- TIME DOMAIN ------------------- #
    # -- We plot the focal plane.
    extractedTemporalFocalPlane = DataAlreadyPostProcessed(dataFolder+"maxIndices.npy")            \
                              and DataAlreadyPostProcessed(dataFolder+"maxValue.npy")              \
                              and DataAlreadyPostProcessed(dataFolder+"focalPointMaxIdxTime.npy")  \
                              and DataAlreadyPostProcessed(dataFolder+"focalPointTime.npy")        \
                              and DataAlreadyPostProcessed(dataFolder+"focalPlaneTime.npy")        \
                              and not args.force

    if (not extractedTemporalFocalPlane):
        maxIndices, maxValue, focalPointMaxIdxTime, focalPointTime, focalPlaneTime = analysis_obj.FindTemporalFocalPlane(analysis_obj.ElectricEnergyDensity, analysis_obj.ElectricEnergyDensity)
        np.save(dataFolder+"maxIndices.npy",     maxIndices)
        np.save(dataFolder+"maxValue.npy",       maxValue)
        np.save(dataFolder+"focalPointTime.npy", focalPointTime)
        np.save(dataFolder+"focalPlaneTime.npy", focalPlaneTime)
        with open(dataFolder+"focalPointMaxIdxTime.npy", 'wb') as fp:
            pickle.dump(focalPointMaxIdxTime, fp)
    else:
        maxIndices     = np.load(dataFolder+"maxIndices.npy")
        maxValue       = np.load(dataFolder+"maxValue.npy")
        focalPointTime = np.load(dataFolder+"focalPointTime.npy")
        focalPlaneTime = np.load(dataFolder+"focalPlaneTime.npy")
        with open(dataFolder+"focalPointMaxIdxTime.npy", 'rb') as fp:
            focalPointMaxIdxTime = pickle.load(fp)

    time_idx = focalPointMaxIdxTime
    z_idx    = maxIndices[focalPointMaxIdxTime][2]

    extractedFocalPlaneTime = DataAlreadyPostProcessed(dataFolder+"ExFocalPlane.npy") \
                          and DataAlreadyPostProcessed(dataFolder+"EyFocalPlane.npy") \
                          and DataAlreadyPostProcessed(dataFolder+"EzFocalPlane.npy") \
                          and DataAlreadyPostProcessed(dataFolder+"BxFocalPlane.npy") \
                          and DataAlreadyPostProcessed(dataFolder+"ByFocalPlane.npy") \
                          and DataAlreadyPostProcessed(dataFolder+"BzFocalPlane.npy") \
                          and not args.force

    if (not extractedFocalPlaneTime):
        ExFocalPlane, EyFocalPlane, EzFocalPlane, BxFocalPlane, ByFocalPlane, BzFocalPlane \
            = analysis_obj.GetFocalPlaneInTimeCartesian(z_idx)

        np.save(dataFolder+"ExFocalPlane.npy", ExFocalPlane)
        np.save(dataFolder+"EyFocalPlane.npy", EyFocalPlane)
        np.save(dataFolder+"EzFocalPlane.npy", EzFocalPlane)
        np.save(dataFolder+"BxFocalPlane.npy", BxFocalPlane)
        np.save(dataFolder+"ByFocalPlane.npy", ByFocalPlane)
        np.save(dataFolder+"BzFocalPlane.npy", BzFocalPlane)
    else:
        ExFocalPlane = np.load(dataFolder+"ExFocalPlane.npy")
        EyFocalPlane = np.load(dataFolder+"EyFocalPlane.npy")
        EzFocalPlane = np.load(dataFolder+"EzFocalPlane.npy")
        BxFocalPlane = np.load(dataFolder+"BxFocalPlane.npy")
        ByFocalPlane = np.load(dataFolder+"ByFocalPlane.npy")
        BzFocalPlane = np.load(dataFolder+"BzFocalPlane.npy")


    analstrat.PlotAllFieldComponentsOnAPlane(analysis_obj.X_meshgrid/1e-6,
                                             analysis_obj.Y_meshgrid/1e-6,
                                             np.transpose(np.abs(ExFocalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(EyFocalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(EzFocalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(BxFocalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(ByFocalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(BzFocalPlane[:,:,time_idx]))**2,
                                             folderName+"/FocalPlaneTime.pdf",
                                             normalization=True)

    # -- Compute the waist in each direction and determine the ellipticity of the
    # -- E_x component.
    intensity_E_waist_x[idx], intensity_E_waist_y[idx], intensityE_waist = \
      analysis_obj.ComputeBeamWaist(ExFocalPlane[:,:,time_idx]**2,0.5)

    print("Waist in the x direction: {}".format(intensity_E_waist_x[idx]))
    print("Waist in the y direction: {}".format(intensity_E_waist_y[idx]))

    intensity_E_ellipticity[idx] = ComputeEllipcity(intensity_E_waist_x[idx],intensity_E_waist_y[idx])

    print("Ellipticity: {}".format(intensity_E_ellipticity[idx]))

    # -- Compute the focal area.
    electric_intensity = np.abs(ExFocalPlane)**2+np.abs(EyFocalPlane)**2+np.abs(EzFocalPlane)**2
    magnetic_intensity = np.abs(BxFocalPlane)**2+np.abs(ByFocalPlane)**2+np.abs(BzFocalPlane)**2

    electric_focal_area = analysis_obj.ComputeFocalArea(electric_intensity[:,:,time_idx],0.5)/(analysis_obj.wavelength[freq_idx]**2)

    print(electric_focal_area)

    # -- Prepare transverse cuts for all time values.
    x_cut, y_cut, field_xcut, field_ycut = analysis_obj.PrepareTransverseCuts(analysis_obj.X_meshgrid,
                                                                              analysis_obj.Y_meshgrid,
                                                                              electric_intensity[:,:,time_idx])
    x_cut_time      = np.empty((x_cut.shape[0], analysis_obj.size_time))
    y_cut_time      = np.empty((x_cut.shape[0], analysis_obj.size_time))
    field_xcut_time = np.empty((x_cut.shape[0], analysis_obj.size_time))
    field_ycut_time = np.empty((x_cut.shape[0], analysis_obj.size_time))

    for i in range(analysis_obj.size_time):
        x_cut_time[:,i], y_cut_time[:,i], field_xcut_time[:,i],field_ycut_time[:,i] \
          = analysis_obj.PrepareTransverseCuts(analysis_obj.X_meshgrid,
                                               analysis_obj.Y_meshgrid,
                                               electric_intensity[:,:,i])

    # -- Time in fs and x in um, normalized amplitude.
    x_cut_time *= 1e6
    time        = analysis_obj.time[:]*1e15
    field_xcut_time /= np.amax(field_xcut_time)

    # -- Prepare a LogNorm for colours andalpha.
    norm = mpl.colors.PowerNorm(gamma=0.1,vmin=0.0, vmax=np.amax(np.abs(time)))

    # -- Prepare the waterfall plot.
    fig = plt.figure(figsize=(4,4))
    ax  = fig.add_subplot(111, projection='3d')
    ax.view_init(20,110)
    verts = []
    max_field = np.empty((time.size))

    for i in range(analysis_obj.size_time):
        xs = np.concatenate([[x_cut_time[0,i]], x_cut_time[:,i], [x_cut_time[-1,i]]])
        ys = np.concatenate([[0],field_xcut_time[:,i],[0]])
        verts.append(list(zip(xs,ys)))
        max_field[i] = np.amax(field_xcut_time[:,i])
        ax.plot(x_cut_time[:,i],field_xcut_time[:,i], zs=time[-1], zdir='y', zorder=-1, color='C0', alpha=1.0-norm(np.abs(time[i]-time[time_idx])))

 #   def EnvelopeFunction(t,amp,width,power,offset):
 #       return amp*np.exp(-np.power((t-offset)/width,power))

#    popt, pcov = curve_fit(EnvelopeFunction, time, max_field, p0=(1.0,1.0,1.0,0.0))

    poly = PolyCollection(verts, rasterized=True, facecolor=None, edgecolor='k', lw=0.7)
    poly.set_alpha(0.5)
    #ax.scatter(time, max_field, zs=x_cut_time[0,0], zdir='x', zorder=-1)
    #ax.plot(time, EnvelopeFunction(time,*popt), zs=x_cut_time[0,0], zdir='x', zorder=-1)
    #ax.plot(x_cut_time[:,time_idx], field_xcut_time[:,time_idx], zs=time[-1], zdir='y', zorder=-1)
    ax.add_collection3d(poly, zs=time, zdir='y')
    ax.ticklabel_format(style='sci', axis='z',scilimits=(0,0))
    ax.set_xlim3d(x_cut_time.min(), x_cut_time.max())
    ax.set_xlabel(r'$x$ [$\mu m$]')
    ax.set_ylim3d(np.amin(time), np.amax(time))
    ax.set_ylabel('Time (fs)')
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_zlim3d(0.0, field_xcut_time.max())
    ax.set_zlabel('Amplitude', rotation=90)
    plt.savefig(folderName+"/ElectricIntensityTimeWaterfall.pdf",dpi=500)
    plt.close()

    # -- Further processing of the waterfall figure.
    cmd = 'pdfcrop --margins "35 0 0 0" {0}/ElectricIntensityTimeWaterfall.pdf {0}/ElectricIntensityTimeWaterfall-cropped.pdf'.format(folderName)
    proc = subprocess.call(shlex.split(cmd))
    print("Waterfall plot done.")

    # -- We plot the electric intensity and its x and y-cuts.
    fig = plt.figure(figsize=(4,4))

    # Image at focal plane.
    axPlane  = plt.subplot2grid((3,3), (0,1), colspan=2,rowspan=2)
    im         = axPlane.pcolormesh(analysis_obj.X_meshgrid*1e6,analysis_obj.Y_meshgrid*1e6, np.transpose(electric_intensity[:,:,time_idx]), cmap='viridis', rasterized=True)
    axPlane.set_aspect('equal')
    axPlane.set_xticklabels('', visible=False)
    axPlane.set_yticklabels('', visible=False)

    # x-cut of Ex.
    axXCut     = plt.subplot2grid((3,3), (2,1), colspan=2)
    axXCut.set_xlabel(r"$x$ [\si{\micro\metre}]")
    axXCut.plot(x_cut*1e6,field_ycut)
    plt.locator_params(axis='y', nbins=4)

    # y-cut of Ex
    axYCut     = plt.subplot2grid((3,3), (0,0), rowspan=2)
    axYCut.set_ylabel(r"$y$ [\si{\micro\metre}]")
    axYCut.plot(field_ycut*1e3,y_cut*1e6)
    plt.locator_params(axis='x', nbins=4)

    # Colobar and other decorations.
    box = axPlane.get_position()

    # create color bar
    axColor = plt.axes([box.x0*1.05 + box.width * 1.05, box.y0, 0.01, box.height])
    plt.colorbar(im, cax = axColor, orientation="vertical")
    plt.savefig(folderName+"/ElectricIntensityTime.pdf", bbox_inches='tight')
    plt.close()


    # -- We plot the sagittal plane
    extractedSagittalPlaneTime = DataAlreadyPostProcessed(dataFolder+"ExSagittalPlane.npy") \
                             and DataAlreadyPostProcessed(dataFolder+"EySagittalPlane.npy") \
                             and DataAlreadyPostProcessed(dataFolder+"EzSagittalPlane.npy") \
                             and DataAlreadyPostProcessed(dataFolder+"BxSagittalPlane.npy") \
                             and DataAlreadyPostProcessed(dataFolder+"BySagittalPlane.npy") \
                             and DataAlreadyPostProcessed(dataFolder+"BzSagittalPlane.npy") \
                             and not args.force

    if (not extractedSagittalPlaneTime):
        ExSagittalPlane, EySagittalPlane, EzSagittalPlane, \
        BxSagittalPlane, BySagittalPlane, BzSagittalPlane   \
            = analysis_obj.GetSagittalPlaneInTimeCartesian()

        np.save(dataFolder+"ExSagittalPlane.npy", ExSagittalPlane)
        np.save(dataFolder+"EySagittalPlane.npy", EySagittalPlane)
        np.save(dataFolder+"EzSagittalPlane.npy", EzSagittalPlane)
        np.save(dataFolder+"BxSagittalPlane.npy", BxSagittalPlane)
        np.save(dataFolder+"BySagittalPlane.npy", BySagittalPlane)
        np.save(dataFolder+"BzSagittalPlane.npy", BzSagittalPlane)
    else:
        ExSagittalPlane = np.load(dataFolder+"ExSagittalPlane.npy")
        EySagittalPlane = np.load(dataFolder+"EySagittalPlane.npy")
        EzSagittalPlane = np.load(dataFolder+"EzSagittalPlane.npy")
        BxSagittalPlane = np.load(dataFolder+"BxSagittalPlane.npy")
        BySagittalPlane = np.load(dataFolder+"BySagittalPlane.npy")
        BzSagittalPlane = np.load(dataFolder+"BzSagittalPlane.npy")


    analstrat.PlotAllFieldComponentsOnAPlane(analysis_obj.R_axial_meshgrid/1e-6,
                                             analysis_obj.Z_axial_meshgrid/1e-6,
                                             np.transpose(np.abs(ExSagittalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(EySagittalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(EzSagittalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(BxSagittalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(BySagittalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(BzSagittalPlane[:,:,time_idx]))**2,
                                             folderName+"/SagittalPlaneTime.pdf",
                                             normalization=True)

    # -- We plot and analyze the meridional plane.
    extractedMeridionalPlaneTime = DataAlreadyPostProcessed(dataFolder+"ExMeridionalPlane.npy") \
                               and DataAlreadyPostProcessed(dataFolder+"EyMeridionalPlane.npy") \
                               and DataAlreadyPostProcessed(dataFolder+"EzMeridionalPlane.npy") \
                               and DataAlreadyPostProcessed(dataFolder+"BxMeridionalPlane.npy") \
                               and DataAlreadyPostProcessed(dataFolder+"ByMeridionalPlane.npy") \
                               and DataAlreadyPostProcessed(dataFolder+"BzMeridionalPlane.npy") \
                               and not args.force

    if (not extractedMeridionalPlaneTime):
        ExMeridionalPlane, EyMeridionalPlane, EzMeridionalPlane, \
        BxMeridionalPlane, ByMeridionalPlane, BzMeridionalPlane   \
            = analysis_obj.GetMeridionalPlaneInTimeCartesian()

        np.save(dataFolder+"ExMeridionalPlane.npy", ExMeridionalPlane)
        np.save(dataFolder+"EyMeridionalPlane.npy", EyMeridionalPlane)
        np.save(dataFolder+"EzMeridionalPlane.npy", EzMeridionalPlane)
        np.save(dataFolder+"BxMeridionalPlane.npy", BxMeridionalPlane)
        np.save(dataFolder+"ByMeridionalPlane.npy", ByMeridionalPlane)
        np.save(dataFolder+"BzMeridionalPlane.npy", BzMeridionalPlane)

    else:
        ExMeridionalPlane = np.load(dataFolder+"ExMeridionalPlane.npy")
        EyMeridionalPlane = np.load(dataFolder+"EyMeridionalPlane.npy")
        EzMeridionalPlane = np.load(dataFolder+"EzMeridionalPlane.npy")
        BxMeridionalPlane = np.load(dataFolder+"BxMeridionalPlane.npy")
        ByMeridionalPlane = np.load(dataFolder+"ByMeridionalPlane.npy")
        BzMeridionalPlane = np.load(dataFolder+"BzMeridionalPlane.npy")


    analstrat.PlotAllFieldComponentsOnAPlane(analysis_obj.R_axial_meshgrid/1e-6,
                                             analysis_obj.Z_axial_meshgrid/1e-6,
                                             np.transpose(np.abs(ExMeridionalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(EyMeridionalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(EzMeridionalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(BxMeridionalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(ByMeridionalPlane[:,:,time_idx]))**2,
                                             np.transpose(np.abs(BzMeridionalPlane[:,:,time_idx]))**2,
                                             folderName+"/MeridionalPlaneTime.pdf",
                                             normalization=True)

# -- Plot global properties.
# Plot only those that are finite.
ellipse_mask = np.isfinite(intensity_E_ellipticity)

fig = plt.figure(figsize=(4,3))
ax  = fig.add_subplot(111)

ax.plot(alpha_f[ellipse_mask],intensity_E_ellipticity[ellipse_mask])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Ellipticity $e$", rotation='horizontal', ha='left', va='bottom')
ax.yaxis.set_label_coords(0.00, 1.05)


plt.savefig(bin_folder+"/Ex_ellipticity.pdf", bbox_inches='tight', dpi=500)
plt.close()

ratio_mask = np.isfinite(max_EzEx)

fig = plt.figure(figsize=(4,3))
ax  = fig.add_subplot(111)

ax.plot(alpha_f[ratio_mask], max_EzEx[ratio_mask])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Ratio of $E_x$ to $E_z$", rotation='horizontal', ha='left')
ax.yaxis.set_label_coords(0.00, 1.05)

plt.savefig(bin_folder+"/ExEzRatio.pdf", bbox_inches='tight', dpi=500)