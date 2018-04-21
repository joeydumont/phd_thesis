# --------------------------------------------------------------------------- #
# Author:          François Fillion-Gourdeau  <francois.fillion@emt.inrs.ca>  #
# Author:          Joey Dumont                <joey.dumont@gmail.com>         #
# Date created:    Apr. 21st, 2018                                            #
# Description:     Created as a test of the StrattoCalculator.                #
#                  In this version, we simply compute the integrands on the   #
#                  mirror for different observation distances.                #
# License:         CC0                                                        #
#                  <https://creativecommons.org/publicdomain/zero/1.0>        #
# --------------------------------------------------------------------------- #

# --------------------------- Modules Importation --------------------------- #4
# -- Plotting
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt
from matplotlib import cm

# -- Analysis
import numpy as np
import scipy as sp
from scipy import integrate

# -- OS and other stuff
import time
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

# --------------------------- Constants Definition -------------------------- #

time_begin = time.clock()

# GLOBAL CONSTANTS
LIGHT_VELOCITY = 299792458;
ALPHA = 1.0/137.035999074;
UNIT_TIME = 1.2880885e-21;
UNIT_LENGTH = 3.86159e-13;
UNIT_MASS = 9.1093829140e-31;
UNIT_ENERGY = UNIT_MASS*LIGHT_VELOCITY*LIGHT_VELOCITY;
COUPLING_QED = 4.0*ALPHA*ALPHA/45.;

# Parameters of the parabola
focal_length = 0.04375/UNIT_LENGTH
Rmin = 0.014/UNIT_LENGTH
Rmax = 0.0875/UNIT_LENGTH

# Parameters of the incident beam
w0 = 0.075/UNIT_LENGTH
energy_per_pulse = 1.0/UNIT_ENERGY
wavelength = 780e-9/UNIT_LENGTH
wavelength1 = 820e-9/UNIT_LENGTH
freq = 2*np.pi/wavelength
freq1 = 2*np.pi/wavelength1
tau = np.pi/np.abs(freq1-freq);
k = freq
E0 = k*np.sqrt(2*energy_per_pulse/tau/np.pi)

# Domain of reflected field
rmin = 0.0
rmax = 1e-6/UNIT_LENGTH
zmin = -1e-6/UNIT_LENGTH
zmax = 1e-6/UNIT_LENGTH
n_points_r = 50
n_points_z = 100

# --------------------------- Function Definition --------------------------- #
# Return normal r component to the parabola
def Normal(r):
	return -0.5*r/focal_length
def UnitNormal(r):
	temp = np.sqrt(4*focal_length**2 + r**2)
	return -r/temp,2*focal_length/temp

# Return incident field for Gaussian TM01 field
def IncidentEr(r):
	return -E0*2*r/k/w0**2*np.exp(-r**2/w0**2)
def IncidentEz(r):
	return 1j*E0*4/k**2/w0**2*(1-r**2/w0**2)*np.exp(-r**2/w0**2)
def IncidentBth(r):
	return E0*2*r/k/w0**2*np.exp(-r**2/w0**2)

# Parabola
def GetZ(r):
	return 0.25*r**2/focal_length - focal_length

# Green's function 
def ComputeU(r,z,r_S,th_S,z_S):
	return np.sqrt(r**2 + r_S**2 - 2*r*r_S*np.cos(th_S) + (z_S-z)**2)
def GreenFunction(r,z,r_S,th_S,z_S):
	u = ComputeU(r,z,r_S,th_S,z_S)
	return np.exp(1j*k*(u-z_S))/u
def GradGreenFunction(r,z,r_S,th_S,z_S):
	u = ComputeU(r,z,r_S,th_S,z_S)
	pref = (1j*k-1/u)/u
	return pref*(r_S*np.cos(th_S) -r), pref*r_S*np.sin(th_S), pref*(z_S-z)
def GradGreenFunctionR(r,z,r_S,th_S,z_S):
	u = ComputeU(r,z,r_S,th_S,z_S)
	pref = (1j*k-1/u)/u
	return pref*(r_S*np.cos(th_S) -r)
def GradGreenFunctionTh(r,z,r_S,th_S,z_S):
	u = ComputeU(r,z,r_S,th_S,z_S)
	pref = (1j*k-1/u)/u
	return pref*r_S*np.sin(th_S)
def GradGreenFunctionZ(r,z,r_S,th_S,z_S):
	u = ComputeU(r,z,r_S,th_S,z_S)
	pref = (1j*k-1/u)/u
	return pref*(z_S-z)	

# Return integrand for surface integrals
def integrandEr(r_S,th_S,r,z):
	z_S = GetZ(r_S)
	Er = IncidentEr(r_S)
	Ez = IncidentEz(r_S)
	Bth = IncidentBth(r_S)
	Nr = Normal(r_S)
	Nz = 1
	NcrossB = -Bth*Nz*np.cos(th_S)
	NdotE = Er*Nr+Ez*Nz
	temp1 = 1j*k*NcrossB
	temp2 = NdotE*GradGreenFunctionR(r,z,r_S,th_S,z_S)
	return 0.5/np.pi*(temp1 + temp2)*GreenFunction(r,z,r_S,th_S,z_S)*r_S

def integrandEz(r_S,th_S,r,z):
	z_S = GetZ(r_S)
	Er = IncidentEr(r_S)
	Ez = IncidentEz(r_S)
	Bth = IncidentBth(r_S)
	Nr = Normal(r_S)
	Nz = 1
	NcrossB = Bth*Nr
	NdotE = Er*Nr+Ez*Nz
	temp1 = 1j*k*NcrossB
	temp2 = NdotE*GradGreenFunctionZ(r,z,r_S,th_S,z_S)
	return 0.5/np.pi*(temp1 + temp2)*GreenFunction(r,z,r_S,th_S,z_S)*r_S

def integrandBth(r_S,th_S,r,z):
	z_S = GetZ(r_S)
	Er = IncidentEr(r_S)
	Ez = IncidentEz(r_S)
	Bth = IncidentBth(r_S)
	Nr = Normal(r_S)
	Nz = 1
	Gr,Gth,Gz = GradGreenFunction(r,z,r_S,th_S,z_S)
	NcrossBcrossG = -Bth*Gth*Nr*np.sin(th_S)+Bth*np.cos(th_S)*(Gz*Nz+Gr*Nr)
	return 0.5/np.pi*NcrossBcrossG*GreenFunction(r,z,r_S,th_S,z_S)*r_S

# Real part of the integrand
def real_integrandEr(r_S,th_S,r,z):
	temp = integrandEr(r_S,th_S,r,z)
	return temp.real
def real_integrandEz(r_S,th_S,r,z):
	temp = integrandEz(r_S,th_S,r,z)
	return temp.real
def real_integrandBth(r_S,th_S,r,z):
	temp = integrandBth(r_S,th_S,r,z)
	return temp.real
# Imaginary part of the integrand
def imag_integrandEr(r_S,th_S,r,z):
	temp = integrandEr(r_S,th_S,r,z)
	return temp.imag
def imag_integrandEz(r_S,th_S,r,z):
	temp = integrandEz(r_S,th_S,r,z)
	return temp.imag
def imag_integrandBth(r_S,th_S,r,z):
	temp = integrandBth(r_S,th_S,r,z)
	return temp.imag

# Return integrand for line integrals
def lineIntegrandEr(th_S,r,z):
	z_S = GetZ(Rmax)
	Bth = IncidentBth(Rmax)
	#nr,nz = UnitNormal(Rmax)
	NcrossNcrossB = -Bth*np.cos(th_S)	
	return 1j*0.5/np.pi/k*NcrossNcrossB*GradGreenFunctionR(r,z,Rmax,th_S,z_S)*GreenFunction(r,z,Rmax,th_S,z_S)*Rmax

def lineIntegrandEz(th_S,r,z):
	z_S = GetZ(Rmax)
	Bth = IncidentBth(Rmax)
	#nr,nz = UnitNormal(Rmax)
	NcrossNcrossB = -Bth*np.cos(th_S)	
	return 1j*0.5/np.pi/k*NcrossNcrossB*GradGreenFunctionZ(r,z,Rmax,th_S,z_S)*GreenFunction(r,z,Rmax,th_S,z_S)*Rmax

# Real part of the integrand
def real_lineIntegrandEr(th_S,r,z):
	temp = lineIntegrandEr(th_S,r,z)
	return temp.real
def real_lineIntegrandEz(th_S,r,z):
	temp = lineIntegrandEz(th_S,r,z)
	return temp.real
def real_lineIntegrandBth(th_S,r,z):
	temp = lineIntegrandBth(th_S,r,z)
	return temp.real
# Imaginary part of the integrand
def imag_lineIntegrandEr(th_S,r,z):
	temp = lineIntegrandEr(th_S,r,z)
	return temp.imag
def imag_lineIntegrandEz(th_S,r,z):
	temp = lineIntegrandEz(th_S,r,z)
	return temp.imag
def imag_lineIntegrandBth(th_S,r,z):
	temp = lineIntegrandBth(th_S,r,z)
	return temp.imag

# Functions that compute the field
def ComputeEr(R,Z):
	real_surface = integrate.nquad(real_integrandEr,[[Rmin, Rmax],[0, 2*np.pi]],args=(R,Z))
	imag_surface = integrate.nquad(imag_integrandEr,[[Rmin, Rmax],[0, 2*np.pi]],args=(R,Z))
	real_line = integrate.nquad(real_lineIntegrandEr,[[Rmin, 2*np.pi]],args=(R,Z))
	imag_line = integrate.nquad(imag_lineIntegrandEr,[[Rmin, 2*np.pi]],args=(R,Z))
	tot_real = real_surface[0] + real_line[0]
	tot_imag = imag_surface[0] + imag_line[0]
	return tot_real,tot_imag

def ComputeEz(R,Z):
	real_surface = integrate.nquad(real_integrandEz,[[Rmin, Rmax],[0, 2*np.pi]],args=(R,Z))
	imag_surface = integrate.nquad(imag_integrandEz,[[Rmin, Rmax],[0, 2*np.pi]],args=(R,Z))
	real_line = integrate.nquad(real_lineIntegrandEz,[[Rmin, 2*np.pi]],args=(R,Z))
	imag_line = integrate.nquad(imag_lineIntegrandEz,[[Rmin, 2*np.pi]],args=(R,Z))
	tot_real = real_surface[0] + real_line[0]
	tot_imag = imag_surface[0] + imag_line[0]
	return tot_real,tot_imag

def ComputeBth(R,Z):
	real_surface = integrate.nquad(real_integrandBth,[[Rmin, Rmax],[0, 2*np.pi]],args=(R,Z))
	imag_surface = integrate.nquad(imag_integrandBth,[[Rmin, Rmax],[0, 2*np.pi]],args=(R,Z))
	tot_real = real_surface[0]
	tot_imag = imag_surface[0]
	return tot_real,tot_imag

# ------------------------------ MAIN FUNCTION ------------------------------ #
r_reflected = np.linspace(rmin,rmax,n_points_r)
z_reflected = np.linspace(zmin,zmax,n_points_z)
R_reflected,Z_reflected = np.meshgrid(r_reflected,z_reflected)

Er_real  = np.zeros((n_points_r,n_points_z))
Ez_real  = np.zeros((n_points_r,n_points_z))
Bth_real = np.zeros((n_points_r,n_points_z))
Er_imag  = np.zeros((n_points_r,n_points_z))
Ez_imag  = np.zeros((n_points_r,n_points_z))
Bth_imag = np.zeros((n_points_r,n_points_z))

# -- Compute the reflected field everywhere on the lattice
#for i in range(n_points_r):
#	for j in range(n_points_z):
#		Er_real[i,j],Er_imag[i,j]    = ComputeEr(r_reflected[i],z_reflected[j])
#		Ez_real[i,j],Ez_imag[i,j]    = ComputeEz(r_reflected[i],z_reflected[j])
#		Bth_real[i,j],Bth_imag[i,j]  = ComputeBth(r_reflected[i],z_reflected[j])


time_end = time.clock()
print("Computation time = {:f} s".format(time_end - time_begin))

# --  Plot the field
#plt.figure()
#plt.title('$E_{r}$')
#plt.xlabel('$z$ (nm)')
#plt.ylabel('$r$ (nm)')
#plt.axes().set_aspect('equal')
#plt.pcolor(z_reflected*UNIT_LENGTH,r_reflected*UNIT_LENGTH,Er_real)
#plt.colorbar()
#plt.savefig("Er-0.04375.pdf", bbox_inches='tight')

#plt.figure()
#plt.title('$E_{z}$')
#plt.xlabel('$z$ (nm)')
#plt.ylabel('$r$ (nm)')
#plt.axes().set_aspect('equal')
#plt.pcolor(z_reflected*UNIT_LENGTH,r_reflected*UNIT_LENGTH,Ez_real)
#plt.colorbar()
#plt.savefig("Ez-0.04375.pdf", bbox_inches='tight')

#plt.figure()
#plt.title(r'$B_{\theta}$')
#plt.xlabel('$z$ (nm)')
#plt.ylabel('$r$ (nm)')
#plt.axes().set_aspect('equal')
#plt.pcolor(z_reflected*UNIT_LENGTH,r_reflected*UNIT_LENGTH,Bth_real)
#plt.colorbar()
#plt.savefig("Bth-0.04375.pdf", bbox_inches='tight')

# -- Compute the integrand on the whole mirror (for phase cancellation).
r_mirror = np.linspace(0,Rmax,200)
theta_mirror = np.linspace(0,2*np.pi,200)
R_mir, TH_mir = np.meshgrid(r_mirror,theta_mirror)
R_mir_plot, Th_mir_plot = np.meshgrid(r_mirror*UNIT_LENGTH,theta_mirror)

zp = np.linspace(0,100*wavelength,11)

for i in range(11):
        phase_focus = integrandEr(R_mir,TH_mir,0,zp[i])

        figPhaseFocus = plt.figure(figsize=(3,2))
        axPhaseFocus  = figPhaseFocus.add_subplot(111)
        #axPhaseFocus.spines['top'].set_visible(False)
        #axPhaseFocus.spines['right'].set_visible(False)
        axPhaseFocus.xaxis.set_ticks_position('bottom')
        axPhaseFocus.yaxis.set_ticks_position('left')
        axPhaseFocus.set_xlabel(r"Radial position on the mirror")
        axPhaseFocus.set_ylabel("Angular position on the mirror",rotation='horizontal', ha='left')
        axPhaseFocus.yaxis.set_label_coords(0.01,1.01)
        axPhaseFocus.set_xlim((0,Rmax*UNIT_LENGTH))
        axPhaseFocus.set_ylim((0,2*np.pi))
        axPhaseFocus.text(0.95,0.85,"$L={:1.0f}\lambda$".format(zp[i]/wavelength), ha='right',transform=axPhaseFocus.transAxes, fontsize=12, color='w',bbox=dict(facecolor='k', alpha=0.5))

        plt.pcolormesh(R_mir_plot,Th_mir_plot, np.real(phase_focus),cmap=cm.inferno, rasterized=True)
        plt.savefig("phase_{:1.0f}L.pdf".format(zp[i]/wavelength), bbox_inches='tight', dpi=500)

#plt.savefig("phase_focus.pdf", bbox_inches='tight')


# figPhase10L = plt.figure(figsize=(6,4))
# axPhase10L  = figPhase10L.add_subplot(111)
# axPhase10L.spines['top'].set_visible(False)
# axPhase10L.spines['right'].set_visible(False)
# axPhase10L.xaxis.set_ticks_position('bottom')
# axPhase10L.yaxis.set_ticks_position('left')
# axPhase10L.set_xlabel(r"Radial position on the mirror")
# axPhase10L.set_ylabel(r"Angular position on the mirror")
# axPhase10L.set_xlim((0,Rmax))
# axPhase10L.set_ylim((0,2*np.pi))

# plt.pcolormesh(R_mir,TH_mir, np.real(phase_10l),cmap=cm.coolwarm, rasterized=True)

# plt.savefig("phase_10L.pdf", bbox_inches='tight')

# figPhase100L = plt.figure(figsize=(6,4))
# axPhase100L  = figPhase100L.add_subplot(111)
# axPhase100L.spines['top'].set_visible(False)
# axPhase100L.spines['right'].set_visible(False)
# axPhase100L.xaxis.set_ticks_position('bottom')
# axPhase100L.yaxis.set_ticks_position('left')
# axPhase100L.set_xlabel(r"Radial position on the mirror")
# axPhase100L.set_ylabel(r"Angular position on the mirror")
# axPhase100L.set_xlim((0,Rmax))
# axPhase100L.set_ylim((0,2*np.pi))

# plt.pcolormesh(R_mir,TH_mir, np.real(phase_100l),cmap=cm.coolwarm, rasterized=True)

# plt.savefig("phase_100L.pdf", bbox_inches='tight')

plt.show()

