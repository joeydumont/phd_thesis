# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created:      Nov. 21st, 2016                                               #
# Description:  We compute the fields in the focal spot with Richards-Wolf.   #
#               Monochromatic fields.                                         #
# Dependencies: - NumPy                                                       #
#               - SciPy                                                       #
#               - H5Py                                                        #
#               - Matplotlib                                                  #
# --------------------------------------------------------------------------- #

# --------------------------- Modules Importation --------------------------- #
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as integration

# ------------------------------ Configuration ------------------------------ #
# -- LaTeX
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'

# -- Fonts
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['font.family'] = 'serif'

# -- Plots
matplotlib.rcParams['legend.numpoints'] = 5
matplotlib.rcParams['figure.figsize'] = '4,2'
matplotlib.rcParams['axes.grid'] = True

# -------------------------------- Constants -------------------------------- #
UNIT_LENGTH    = 3.86159e-13
UNIT_TIME      = 1.2880885e-21
SPEED_OF_LIGHT = 299792458
EPSILON_0      = 8.85418782e-12
MU_0           = 4*np.pi*1.0e-7
UNIT_E_FIELD   = 1.3e18
UNIT_B_FIELD   = UNIT_E_FIELD/SPEED_OF_LIGHT
UNIT_MASS      = 9.109382914e-31

# ------------------------------ MAIN FUNCTION ------------------------------- #

# ---------------------------- Plot Richards-Wolf ---------------------------- #
# -- Load the data.
x_RW_fp  = np.loadtxt("x.txt")
y_RW_fp  = np.loadtxt("y.txt")
ExRW_fp  = -np.genfromtxt("Ex_sed.txt",dtype=complex)
EyRW_fp  = -np.genfromtxt("Ey_sed.txt",dtype=complex)
EzRW_fp  = -np.genfromtxt("Ez_sed.txt",dtype=complex)
BxRW_fp  = -np.genfromtxt("Bx_sed.txt",dtype=complex)
ByRW_fp  = -np.genfromtxt("By_sed.txt",dtype=complex)
BzRW_fp  = -np.genfromtxt("Bz_sed.txt",dtype=complex)

ExRW_fp  = np.abs(ExRW_fp)*np.cos(np.angle(ExRW_fp))
EyRW_fp  = np.abs(EyRW_fp)*np.cos(np.angle(EyRW_fp))
EzRW_fp  = np.abs(EzRW_fp)*np.sin(np.angle(EzRW_fp))
BxRW_fp  = np.abs(BxRW_fp)*np.cos(np.angle(BxRW_fp))
ByRW_fp  = np.abs(ByRW_fp)*np.cos(np.angle(ByRW_fp))
BzRW_fp  = np.abs(BzRW_fp)*np.sin(np.angle(BzRW_fp))

maxField_fp = np.amax(np.abs(ExRW_fp))
r      = np.linspace(0.0,3.0e-6/UNIT_LENGTH,100)
th     = np.linspace(0.0,2*np.pi,100)
R, Th  = np.meshgrid(r,th)
X_RW      = R*np.cos(Th)*UNIT_LENGTH
Y_RW      = R*np.sin(Th)*UNIT_LENGTH

color_opt = {'cmap':'viridis', 'rasterized':True}

# - - Plot the fields.
figRW_fp = plt.figure(figsize=(8,3))
figRW_fp.subplots_adjust(wspace=0.1,hspace=0.5)

axExRW_fp = plt.subplot2grid((2,3), (0,0))
im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,np.transpose(np.real(ExRW_fp))/maxField_fp, **color_opt,vmin=-1.0,vmax=0.3)
axExRW_fp.set_ylabel(r"$y$ [\si{\micro\metre}]")
axExRW_fp.set_aspect('equal')
axExRW_fp.set_title(r"$E_x$")
plt.colorbar()

axEyRW_fp = plt.subplot2grid((2,3), (0,1))
im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,np.transpose(np.real(EyRW_fp))/maxField_fp, **color_opt,vmin=-0.15, vmax=0.15)
axEyRW_fp.set_aspect('equal')
axEyRW_fp.set_title(r"$E_y$")
plt.colorbar()

axEzRW_fp = plt.subplot2grid((2,3), (0,2))
im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,np.transpose(np.real(EzRW_fp))/maxField_fp, **color_opt, vmin=-0.6,vmax=0.6)
axEzRW_fp.set_aspect('equal')
axEzRW_fp.set_title(r"$E_z$")
plt.colorbar()

axBxRW_fp = plt.subplot2grid((2,3), (1,0))
im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,np.transpose(np.real(BxRW_fp))/maxField_fp, **color_opt, vmin=-0.1,vmax=0.1)
axBxRW_fp.set_aspect('equal')
axBxRW_fp.set_ylabel(r"$y$ [\si{\micro\metre}]")
axBxRW_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")
axBxRW_fp.set_title(r"$B_x$")

plt.colorbar()

axByRW_fp = plt.subplot2grid((2,3), (1,1))
im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,np.transpose(np.real(ByRW_fp))/maxField_fp, **color_opt,vmin=-1.0, vmax=0.2)
axByRW_fp.set_aspect('equal')
axByRW_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")
axByRW_fp.set_title(r"$B_y$")
plt.colorbar()

axBzRW_fp = plt.subplot2grid((2,3), (1,2))
im        = plt.pcolormesh(X_RW/1e-6,Y_RW/1e-6,np.transpose(np.real(BzRW_fp))/maxField_fp, **color_opt, vmin=-0.6, vmax=0.6)
axBzRW_fp.set_aspect('equal')
axBzRW_fp.set_title(r"$B_z$")
axBzRW_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")

plt.colorbar()

plt.savefig("RichardsWolf_fp.pdf", bbox_inches='tight', dpi=500)

# ----------------------------- Plot Stratton-Chu ---------------------------- #
# -- Load the data.
X_ST     = np.loadtxt("X.dat")
Y_ST     = np.loadtxt("Y.dat")
#X_ST,Y_ST= np.meshgrid(x_ST_fp,y_ST_fp)
ExST_fp  = np.genfromtxt("Ex_fp_f_st.txt",dtype=complex)
EyST_fp  = np.genfromtxt("Ey_fp_f_st.txt",dtype=complex)
EzST_fp  = np.genfromtxt("Ez_fp_f_st.txt",dtype=complex)
BxST_fp  = np.genfromtxt("Bx_fp_f_st.txt",dtype=complex)
ByST_fp  = np.genfromtxt("By_fp_f_st.txt",dtype=complex)
BzST_fp  = np.genfromtxt("Bz_fp_f_st.txt",dtype=complex)

ExST_fp  = np.abs(ExST_fp)*np.cos(np.angle(ExST_fp))
EyST_fp  = np.abs(EyST_fp)*np.cos(np.angle(EyST_fp))
EzST_fp  = np.abs(EzST_fp)*np.sin(np.angle(EzST_fp))
BxST_fp  = np.abs(BxST_fp)*np.cos(np.angle(BxST_fp))
ByST_fp  = np.abs(ByST_fp)*np.cos(np.angle(ByST_fp))
BzST_fp  = np.abs(BzST_fp)*np.sin(np.angle(BzST_fp))

maxField_fp = np.amax(np.abs(ExST_fp))

# -- Plot the fields.
figST_fp = plt.figure(figsize=(8,3))
figST_fp.subplots_adjust(wspace=0.1,hspace=0.5)

axExST_fp = plt.subplot2grid((2,3), (0,0))
im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,np.transpose(np.real(ExST_fp))/maxField_fp, **color_opt,vmin=-1.0,vmax=0.3)
axExST_fp.set_ylabel(r"$y$ [\si{\micro\metre}]")
axExST_fp.set_aspect('equal')
axExST_fp.set_title(r"$E_x$")
plt.colorbar()

axEyST_fp = plt.subplot2grid((2,3), (0,1))
im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,np.transpose(np.real(EyST_fp))/maxField_fp, **color_opt,vmin=-0.15, vmax=0.15)
axEyST_fp.set_aspect('equal')
axEyST_fp.set_title(r"$E_y$")
plt.colorbar()

axEzST_fp = plt.subplot2grid((2,3), (0,2))
im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,np.transpose(np.real(EzST_fp))/maxField_fp, **color_opt, vmin=-0.6,vmax=0.6)
axEzST_fp.set_aspect('equal')
axEzST_fp.set_title(r"$E_z$")
plt.colorbar()

axBxST_fp = plt.subplot2grid((2,3), (1,0))
im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,np.transpose(np.real(BxST_fp))/maxField_fp, **color_opt, vmin=-0.1,vmax=0.1)
axBxST_fp.set_aspect('equal')
axBxST_fp.set_ylabel(r"$y$ [\si{\micro\metre}]")
axBxST_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")
axBxST_fp.set_title(r"$B_x$")

plt.colorbar()

axByST_fp = plt.subplot2grid((2,3), (1,1))
im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,np.transpose(np.real(ByST_fp))/maxField_fp, **color_opt,vmin=-1.0, vmax=0.2)
axByST_fp.set_aspect('equal')
axByST_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")
axByST_fp.set_title(r"$B_y$")
plt.colorbar()

axBzST_fp = plt.subplot2grid((2,3), (1,2))
im        = plt.pcolormesh(X_ST/1e-6,Y_ST/1e-6,np.transpose(np.real(BzST_fp))/maxField_fp, **color_opt, vmin=-0.6, vmax=0.6)
axBzST_fp.set_aspect('equal')
axBzST_fp.set_title(r"$B_z$")
axBzST_fp.set_xlabel(r"$x$ [\si{\micro\metre}]")

plt.colorbar()

plt.savefig("StrattonChu_fp.pdf", bbox_inches='tight', dpi=500)

# # -- Plot the axial plane.
# z_RW_ap  = np.loadtxt("z.txt")
# R, Z     = np.meshgrid(r,z_RW_ap)
# R        *=UNIT_LENGTH
# Z        *=UNIT_LENGTH
# ExRW_ap  = -np.genfromtxt("Ex_axial_sed.txt",dtype=complex)
# EyRW_ap  = -np.genfromtxt("Ey_axial_sed.txt",dtype=complex)
# EzRW_ap  = -np.genfromtxt("Ez_axial_sed.txt",dtype=complex)
# BxRW_ap  = -np.genfromtxt("Bx_axial_sed.txt",dtype=complex)
# ByRW_ap  = -np.genfromtxt("By_axial_sed.txt",dtype=complex)
# BzRW_ap  = -np.genfromtxt("Bz_axial_sed.txt",dtype=complex)

# ExRW_ap  = np.abs(ExRW_ap)*np.cos(np.angle(ExRW_ap))
# EyRW_ap  = np.abs(EyRW_ap)*np.cos(np.angle(EyRW_ap))
# EzRW_ap  = np.abs(EzRW_ap)*np.sin(np.angle(EzRW_ap))
# BxRW_ap  = np.abs(BxRW_ap)*np.cos(np.angle(BxRW_ap))
# ByRW_ap  = np.abs(ByRW_ap)*np.cos(np.angle(ByRW_ap))
# BzRW_ap  = np.abs(BzRW_ap)*np.sin(np.angle(BzRW_ap))

# maxField_ap = np.amax(np.abs(np.real(ExRW_ap)))

# figRW_ap = plt.figure()

# axExRW_ap = plt.subplot2grid((2,3), (0,0))
# im        = plt.pcolormesh(Z/1e-6,R/1e-6,np.transpose(np.real(ExRW_ap))/maxField_fp, **color_opt)
# axExRW_ap.set_aspect('equal')
# plt.colorbar()

# axEyRW_ap = plt.subplot2grid((2,3), (0,1))
# im        = plt.pcolormesh(Z/1e-6,R/1e-6,np.transpose(np.real(EyRW_ap))/maxField_fp, **color_opt)
# axEyRW_ap.set_aspect('equal')
# plt.colorbar()

# axEzRW_ap = plt.subplot2grid((2,3), (0,2))
# im        = plt.pcolormesh(Z/1e-6,R/1e-6,np.transpose(np.real(EzRW_ap))/maxField_fp, **color_opt)
# axEzRW_ap.set_aspect('equal')
# plt.colorbar()

# axBxRW_ap = plt.subplot2grid((2,3), (1,0))
# im        = plt.pcolormesh(Z/1e-6,R/1e-6,np.transpose(np.real(BxRW_ap))/maxField_fp, **color_opt)
# axBxRW_ap.set_aspect('equal')
# plt.colorbar()

# axByRW_ap = plt.subplot2grid((2,3), (1,1))
# im        = plt.pcolormesh(Z/1e-6,R/1e-6,np.transpose(np.real(ByRW_ap))/maxField_fp, **color_opt)
# axByRW_ap.set_aspect('equal')
# plt.colorbar()

# axBzRW_ap = plt.subplot2grid((2,3), (1,2))
# im        = plt.pcolormesh(Z/1e-6,R/1e-6,np.transpose(np.real(BzRW_ap))/maxField_fp, **color_opt)
# axBzRW_ap.set_aspect('equal')
# plt.colorbar()


# plt.tight_layout()

# plt.savefig("RichardsWolf_ap.pdf", bbox_inches='tight', dpi=500)