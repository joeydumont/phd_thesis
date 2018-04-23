# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created:      Apr. 4th, 2018                                                #
# Description:  We plot the fields computed via the Richards-Wolf formalism   #
#               via our FastRW implementation.                                #
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
AnalysisStratto = imp.load_source('AnalysisStratto', "../../python-tools/AnalysisStrattoCalculator.py")
import vphys
import AnalysisStratto as analstrat


# ------------------------------ Configuration ------------------------------ #
# -- LaTeX

#-- We reset the LaTeX parameters to enable XeLaTeX.
mpl.rcParams.update(vphys.default_pgf_configuration())

# --------------------------- Function Definition --------------------------- #

def LoadComplexData(file,**genfromtext_args):
    """
    Load complex data in the C format in numpy.
    """
    complex_parser = np.vectorize(lambda x: complex(*eval(x)))
    return complex_parser(np.genfromtxt(file, dtype=str,**genfromtext_args))

def FunctionalToPlot(Ex,Ey,Ez,Bx,By,Bz, normalized=True):
    """
    Defines the function of the field to be plotted.
    """
    Ex = np.transpose(np.abs(Ex)**2)
    Ey = np.transpose(np.abs(Ey)**2)
    Ez = np.transpose(np.abs(Ez)**2)
    Bx = np.transpose(np.abs(Bx)**2)
    By = np.transpose(np.abs(By)**2)
    Bz = np.transpose(np.abs(Bz)**2)

    if (normalized):
        maxfield = np.amax([Ex, Ey, Ez, Bx, By, Bz])
        Ex /= maxfield
        Ey /= maxfield
        Ez /= maxfield
        Bx /= maxfield
        By /= maxfield
        Bz /= maxfield

    return Ex,Ey,Ez,Bx,By,Bz

def ComputeCorrelation(r_field,th_field,field,r_ref,th_ref,ref_field, cor_type="pearson"):
    """
    We compute the overlap integral in the focal plane. To do so,
    we use a spline interpolation for both fields, then integrate.
    """
    field_interp = interpolate.RectBivariateSpline(r_field, th_field,field)
    ref_interp   = interpolate.RectBivariateSpline(r_ref, th_ref,ref_field)

    if cor_type == "pearson":

        r = np.linspace(0.0, min(r_field[-1],r_ref[-1]), 100)
        th = np.linspace(0,2*np.pi,100)
        field_interp_values = field_interp(r,th)
        ref_interp_values   = ref_interp(r,th)

        correlation = stats.pearsonr(field_interp_values.flatten(),ref_interp_values.flatten())[0]

    elif cor_type == "absdiff":
        r = np.linspace(0.0, min(r_field[-1],r_ref[-1]), 100)
        th = np.linspace(0,2*np.pi,100)
        field_interp_values = field_interp(r,th)
        ref_interp_values   = ref_interp(r,th)

        correlation = np.abs(field_interp_values-ref_interp_values)

    elif cor_type == "overlap_integral":

        def numeratorIntegrand(r,th):
            return r*field_interp(r,th)*ref_interp(r,th)

        def denominatorIntegrand(r,th):
            return r*ref_interp(r,th)**2

        correlation = integrate.dblquad(numeratorIntegrand, 0, 2*np.pi, lambda x: 0.0, lambda x: min(r_field[-1], r_ref[-1]))[0] \
                     /integrate.dblquad(denominatorIntegrand, 0, 2*np.pi, lambda x: 0.0, lambda x: min(r_field[-1], r_ref[-1]))[0]

    return correlation

# ------------------------------ MAIN FUNCTION ------------------------------- #

# ---------------------------- Plot Richards-Wolf ---------------------------- #

# -- Load data for the loop.
loop_file      = open("focal_na.txt")
loop_lines     = loop_file.read().splitlines()

data_folder_rw = "../../bin/Fields/RichardsWolf/"
data_folder_sc = "../../bin/Fields/Stratto/na_vsf_lin_g/"

# -- Overlap data.
Ex_overlap = np.zeros((len(loop_lines)))
Ey_overlap = np.zeros((len(loop_lines)))
Ez_overlap = np.zeros((len(loop_lines)))
Bx_overlap = np.zeros((len(loop_lines)))
By_overlap = np.zeros((len(loop_lines)))
Bz_overlap = np.zeros((len(loop_lines)))

ExAbsDiffMax = np.zeros((len(loop_lines)))
EyAbsDiffMax = np.zeros((len(loop_lines)))
EzAbsDiffMax = np.zeros((len(loop_lines)))
BxAbsDiffMax = np.zeros((len(loop_lines)))
ByAbsDiffMax = np.zeros((len(loop_lines)))
BzAbsDiffMax = np.zeros((len(loop_lines)))

for idx, i in enumerate(loop_lines):
    # -- Load the data.
    data_folder_rw_loop = data_folder_rw+"f{}/".format(i)
    r_RW_fp  = np.loadtxt(data_folder_rw_loop+"r_fastrw.txt",skiprows=2)
    th_RW_fp = np.loadtxt(data_folder_rw_loop+"th_fastrw.txt",skiprows=2)
    ExRW_fp  = LoadComplexData(data_folder_rw_loop+"/field-component-0.txt",skip_header=2)
    EyRW_fp  = LoadComplexData(data_folder_rw_loop+"/field-component-1.txt",skip_header=2)
    EzRW_fp  = LoadComplexData(data_folder_rw_loop+"/field-component-2.txt",skip_header=2)
    BxRW_fp  = LoadComplexData(data_folder_rw_loop+"/field-component-3.txt",skip_header=2)
    ByRW_fp  = LoadComplexData(data_folder_rw_loop+"/field-component-4.txt",skip_header=2)
    BzRW_fp  = LoadComplexData(data_folder_rw_loop+"/field-component-5.txt",skip_header=2)

    R, Th  = np.meshgrid(r_RW_fp,th_RW_fp)
    X_RW      = R*np.cos(Th)*analstrat.UNIT_LENGTH
    Y_RW      = R*np.sin(Th)*analstrat.UNIT_LENGTH

    ExRW_fp_plot,EyRW_fp_plot,EzRW_fp_plot,BxRW_fp_plot,ByRW_fp_plot,BzRW_fp_plot \
                = FunctionalToPlot(ExRW_fp,EyRW_fp,EzRW_fp,BxRW_fp,ByRW_fp,BzRW_fp)

    color_opt = {'cmap':'viridis', 'rasterized':True}

    # - - Plot the fields.
    figRW_fp = plt.figure(figsize=(8,3))
    figRW_fp.subplots_adjust(wspace=0.1,hspace=0.5)

    axExRW_fp = plt.subplot2grid((2,3), (0,0))
    im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,ExRW_fp_plot, **color_opt)#,vmin=-1.0,vmax=0.3)
    axExRW_fp.set_ylabel(r"$y$ [\si{\micro\metre}]")
    axExRW_fp.set_aspect('equal')
    axExRW_fp.set_title(r"$E_x$")
    plt.colorbar()

    axEyRW_fp = plt.subplot2grid((2,3), (0,1))
    im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,EyRW_fp_plot, **color_opt)#,vmin=-0.15, vmax=0.15)
    axEyRW_fp.set_aspect('equal')
    axEyRW_fp.set_title(r"$E_y$")
    plt.colorbar()

    axEzRW_fp = plt.subplot2grid((2,3), (0,2))
    im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,EzRW_fp_plot, **color_opt)#, vmin=-0.6,vmax=0.6)
    axEzRW_fp.set_aspect('equal')
    axEzRW_fp.set_title(r"$E_z$")
    plt.colorbar()

    axBxRW_fp = plt.subplot2grid((2,3), (1,0))
    im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,BxRW_fp_plot, **color_opt)#, vmin=-0.1,vmax=0.1)
    axBxRW_fp.set_aspect('equal')
    axBxRW_fp.set_ylabel(r"$y$ [\si{\micro\metre}]")
    axBxRW_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")
    axBxRW_fp.set_title(r"$B_x$")

    plt.colorbar()

    axByRW_fp = plt.subplot2grid((2,3), (1,1))
    im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,ByRW_fp_plot, **color_opt)#,vmin=-1.0, vmax=0.2)
    axByRW_fp.set_aspect('equal')
    axByRW_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")
    axByRW_fp.set_title(r"$B_y$")
    plt.colorbar()

    axBzRW_fp = plt.subplot2grid((2,3), (1,2))
    im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,BzRW_fp_plot, **color_opt)#, vmin=-0.6, vmax=0.6)
    axBzRW_fp.set_aspect('equal')
    axBzRW_fp.set_title(r"$B_z$")
    axBzRW_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")

    plt.colorbar()

    plt.savefig(data_folder_rw_loop+"/RichardsWolf_fp.pdf", bbox_inches='tight', dpi=500)
    plt.close()

    # ----------------------------- Plot Stratton-Chu ---------------------------- #
    # -- Load the data.
    data_folder_sc_loop = data_folder_sc+"/stra-vsf-lin_focal{}.BQ/".format(i)
    print(data_folder_sc_loop)
    analysis_obj = analstrat.Analysis3D(freq_field=data_folder_sc_loop+"/Field_reflected.hdf5")
    r_st     = analysis_obj.coord_r[:]*analstrat.UNIT_LENGTH
    th_st    = analysis_obj.coord_theta[:]
    X_ST     = analysis_obj.X_meshgrid#np.loadtxt(data_dir+"X.dat")
    Y_ST     = analysis_obj.Y_meshgrid#np.loadtxt(data_dir+"Y.dat")

    # -- Focal plane
    # We must change the sign of the longitudinal component, as we did not choose
    # the same z axis direction in both codes.
    ErST_fp  = analysis_obj.GetFrequencyComponent("Er",  analysis_obj.size_freq//2)[:,:,analysis_obj.size_z//2]#np.genfromtxt(data_dir+"Ex_fp_f.txt",dtype=complex)
    EtST_fp  = analysis_obj.GetFrequencyComponent("Eth", analysis_obj.size_freq//2)[:,:,analysis_obj.size_z//2]#np.genfromtxt(data_dir+"Ey_fp_f.txt",dtype=complex)
    EzST_fp  = analysis_obj.GetFrequencyComponent("Ez",  analysis_obj.size_freq//2)[:,:,analysis_obj.size_z//2]#np.genfromtxt(data_dir+"Ez_fp_f.txt",dtype=complex)
    BrST_fp  = analysis_obj.GetFrequencyComponent("Br",  analysis_obj.size_freq//2)[:,:,analysis_obj.size_z//2]#np.genfromtxt(data_dir+"Bx_fp_f.txt",dtype=complex)
    BtST_fp  = analysis_obj.GetFrequencyComponent("Bth", analysis_obj.size_freq//2)[:,:,analysis_obj.size_z//2]#np.genfromtxt(data_dir+"By_fp_f.txt",dtype=complex)
    BzST_fp  = analysis_obj.GetFrequencyComponent("Bz",  analysis_obj.size_freq//2)[:,:,analysis_obj.size_z//2]#np.genfromtxt(data_dir+"Bz_fp_f.txt",dtype=complex)

    # -- We compute the Cartesian components of the fields.
    imax, jmax = EzST_fp.shape
    ExST_fp  = np.zeros((imax, jmax),dtype=complex)
    EyST_fp  = np.zeros((imax, jmax),dtype=complex)
    BxST_fp  = np.zeros((imax, jmax),dtype=complex)
    ByST_fp  = np.zeros((imax, jmax),dtype=complex)

    for j in range(imax):
      for k in range(jmax):
        theta = analysis_obj.coord_theta[k]
        c     = np.cos(theta)
        s     = np.sin(theta)

        ExST_fp[j,k] = c*ErST_fp[j,k]-s*EtST_fp[j,k]
        EyST_fp[j,k] = s*ErST_fp[j,k]+c*EtST_fp[j,k]
        BxST_fp[j,k] = c*BrST_fp[j,k]-s*BtST_fp[j,k]
        ByST_fp[j,k] = s*BrST_fp[j,k]+c*BtST_fp[j,k]


    ExST_fp_plot,EyST_fp_plot,EzST_fp_plot,BxST_fp_plot,ByST_fp_plot,BzST_fp_plot \
                = FunctionalToPlot(ExST_fp,EyST_fp,EzST_fp,BxST_fp,ByST_fp,BzST_fp)

    # -- Plot the fields.
    figST_fp = plt.figure(figsize=(8,3))
    figST_fp.subplots_adjust(wspace=0.1,hspace=0.5)

    axExST_fp = plt.subplot2grid((2,3), (0,0))
    im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,ExST_fp_plot, **color_opt)#,vmin=-1.0,vmax=0.3)
    axExST_fp.set_ylabel(r"$y$ [\si{\micro\metre}]")
    axExST_fp.set_aspect('equal')
    axExST_fp.set_title(r"$E_x$")
    plt.colorbar()

    axEyST_fp = plt.subplot2grid((2,3), (0,1))
    im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,EyST_fp_plot, **color_opt)#,vmin=-0.15, vmax=0.15)
    axEyST_fp.set_aspect('equal')
    axEyST_fp.set_title(r"$E_y$")
    plt.colorbar()

    axEzST_fp = plt.subplot2grid((2,3), (0,2))
    im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,EzST_fp_plot, **color_opt)#, vmin=-0.6,vmax=0.6)
    axEzST_fp.set_aspect('equal')
    axEzST_fp.set_title(r"$E_z$")
    plt.colorbar()

    axBxST_fp = plt.subplot2grid((2,3), (1,0))
    im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,BxST_fp_plot, **color_opt)#, vmin=-0.1,vmax=0.1)
    axBxST_fp.set_aspect('equal')
    axBxST_fp.set_ylabel(r"$y$ [\si{\micro\metre}]")
    axBxST_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")
    axBxST_fp.set_title(r"$B_x$")

    plt.colorbar()

    axByST_fp = plt.subplot2grid((2,3), (1,1))
    im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,ByST_fp_plot, **color_opt)#,vmin=-1.0, vmax=0.2)
    axByST_fp.set_aspect('equal')
    axByST_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")
    axByST_fp.set_title(r"$B_y$")
    plt.colorbar()

    axBzST_fp = plt.subplot2grid((2,3), (1,2))
    im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,BzST_fp_plot, **color_opt)#, vmin=-0.6, vmax=0.6)
    axBzST_fp.set_aspect('equal')
    axBzST_fp.set_title(r"$B_z$")
    axBzST_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")
    plt.colorbar()

    plt.savefig(data_folder_sc_loop+"/StrattonChu_fp.pdf", bbox_inches='tight', dpi=500)
    plt.close()

    # ----------------------------- Plot Stratton-Chu ---------------------------- #
    # -- Compute the overlap integral.
    Ex_overlap[idx] = ComputeCorrelation(r_st, th_st, ExST_fp_plot, r_RW_fp, th_RW_fp, ExRW_fp_plot)
    Ey_overlap[idx] = ComputeCorrelation(r_st, th_st, EyST_fp_plot, r_RW_fp, th_RW_fp, EyRW_fp_plot)
    Ez_overlap[idx] = ComputeCorrelation(r_st, th_st, EzST_fp_plot, r_RW_fp, th_RW_fp, EzRW_fp_plot)
    Bx_overlap[idx] = ComputeCorrelation(r_st, th_st, BxST_fp_plot, r_RW_fp, th_RW_fp, BxRW_fp_plot)
    By_overlap[idx] = ComputeCorrelation(r_st, th_st, ByST_fp_plot, r_RW_fp, th_RW_fp, ByRW_fp_plot)
    Bz_overlap[idx] = ComputeCorrelation(r_st, th_st, BzST_fp_plot, r_RW_fp, th_RW_fp, BzRW_fp_plot)

    #intensityRW = Ex
    #IntensityOverlap[idx]

    # -- Compute the absolute difference.
    ExAbsDiff = ComputeCorrelation(r_st, th_st, ExST_fp_plot, r_RW_fp, th_RW_fp, ExRW_fp_plot, cor_type="absdiff")
    EyAbsDiff = ComputeCorrelation(r_st, th_st, EyST_fp_plot, r_RW_fp, th_RW_fp, EyRW_fp_plot, cor_type="absdiff")
    EzAbsDiff = ComputeCorrelation(r_st, th_st, EzST_fp_plot, r_RW_fp, th_RW_fp, EzRW_fp_plot, cor_type="absdiff")
    BxAbsDiff = ComputeCorrelation(r_st, th_st, BxST_fp_plot, r_RW_fp, th_RW_fp, BxRW_fp_plot, cor_type="absdiff")
    ByAbsDiff = ComputeCorrelation(r_st, th_st, ByST_fp_plot, r_RW_fp, th_RW_fp, ByRW_fp_plot, cor_type="absdiff")
    BzAbsDiff = ComputeCorrelation(r_st, th_st, BzST_fp_plot, r_RW_fp, th_RW_fp, BzRW_fp_plot, cor_type="absdiff")

    ExAbsDiffMax[idx] = np.mean(ExAbsDiff)
    EyAbsDiffMax[idx] = np.mean(EyAbsDiff)
    EzAbsDiffMax[idx] = np.mean(EzAbsDiff)
    BxAbsDiffMax[idx] = np.mean(BxAbsDiff)
    ByAbsDiffMax[idx] = np.mean(ByAbsDiff)
    BzAbsDiffMax[idx] = np.mean(BzAbsDiff)

# -- Plot the fields.
focal_length = np.array(loop_lines,dtype=float)
figOverlap = plt.figure(figsize=(8,3))
figOverlap.subplots_adjust(wspace=0.1,hspace=0.5)

axExOverlap = plt.subplot2grid((2,3), (0,0))
im          = plt.plot(focal_length,Ex_overlap)

axEyOverlap = plt.subplot2grid((2,3), (0,1))
im          = plt.plot(focal_length,Ey_overlap)

axEzOverlap = plt.subplot2grid((2,3), (0,2))
im          = plt.plot(focal_length,Ez_overlap)

axBxOverlap = plt.subplot2grid((2,3), (1,0))
im          = plt.plot(focal_length,Bx_overlap)

axByOverlap = plt.subplot2grid((2,3), (1,1))
im          = plt.plot(focal_length,By_overlap)

axBzOverlap = plt.subplot2grid((2,3), (1,2))
im          = plt.plot(focal_length,Bz_overlap)

plt.savefig(data_folder_sc+"/CorrelationFields.pdf")
plt.close()

# -- Plot the fields.
figOverlap = plt.figure(figsize=(8,3))
figOverlap.subplots_adjust(wspace=0.1,hspace=0.5)

axExOverlap = plt.subplot2grid((2,3), (0,0))
im          = plt.plot(focal_length,ExAbsDiffMax)

axEyOverlap = plt.subplot2grid((2,3), (0,1))
im          = plt.plot(focal_length,EyAbsDiffMax)

axEzOverlap = plt.subplot2grid((2,3), (0,2))
im          = plt.plot(focal_length,EzAbsDiffMax)

axBxOverlap = plt.subplot2grid((2,3), (1,0))
im          = plt.plot(focal_length,BxAbsDiffMax)

axByOverlap = plt.subplot2grid((2,3), (1,1))
im          = plt.plot(focal_length,ByAbsDiffMax)

axBzOverlap = plt.subplot2grid((2,3), (1,2))
im          = plt.plot(focal_length,BzAbsDiffMax)

plt.savefig(data_folder_sc+"/AbsDiffFields.pdf")