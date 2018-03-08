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
import time

# ------------------------------ Configuration ------------------------------ #
# -- LaTeX
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{units}'

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

# ----------------------------- Class Definition ---------------------------- #
# -- We define a class that computes the reflected field in the            -- #
# -- Richards-Wolf formalism.                                              -- #
# --------------------------------------------------------------------------- #
def complex_dblquad(func,a,b,f,g):
  def real_function(y,x):
    return np.real(func(y,x))
  def imag_function(y,x):
    return np.imag(func(y,x))

  real_integral = integration.dblquad(real_function,a,b,f,g,epsrel=1.0e-5)
  imag_integral = integration.dblquad(imag_function,a,b,f,g,epsrel=1.0e-5)
  return real_integral[0]+1j*imag_integral[0]

class RichardsWolf:
  """
  This class takes callables that define the Cartesian components of the incident
  electric field, a callable to the apodization factor q(alpha) that defines
  the reflecting surface and the value of alpha_max which defines the aperture
  of the reflecting surface. We also provide the value of the wavenumber.

  It then computes the Cartesian components of the reflected field at provided
  values of r,theta,z in a cylindrical coordinate system.
  """
  def __init__(self,Ex,Ey,q,alpha_max,k):
    """
    Sets the callables and the value of alpha_max.
    """
    self.Ex        = Ex
    self.Ey        = Ey
    self.q         = q
    self.alpha_max = alpha_max
    self.k         = k

  def ComputeEx(self,r,theta,z):
    self.r  = r
    self.th = theta
    self.z  = z

    return complex_dblquad(self.__RW_Ex, 0, self.alpha_max, lambda x: 0.0, lambda x: 2.0*np.pi)

  def ComputeEy(self,r,theta,z):
    self.r  = r
    self.th = theta
    self.z  = z

    return complex_dblquad(self.__RW_Ey, 0, self.alpha_max, lambda x: 0.0, lambda x: 2.0*np.pi)

  def ComputeEz(self,r,theta,z):
    self.r  = r
    self.th = theta
    self.z  = z

    return complex_dblquad(self.__RW_Ez, 0, self.alpha_max, lambda x: 0.0, lambda x: 2.0*np.pi)

  def ComputeBx(self,r,theta,z):
    self.r  = r
    self.th = theta
    self.z  = z

    return complex_dblquad(self.__RW_Bx, 0, self.alpha_max, lambda x: 0.0, lambda x: 2.0*np.pi)

  def ComputeBy(self,r,theta,z):
    self.r  = r
    self.th = theta
    self.z  = z

    return complex_dblquad(self.__RW_By, 0, self.alpha_max, lambda x: 0.0, lambda x: 2.0*np.pi)

  def ComputeBz(self,r,theta,z):
    self.r  = r
    self.th = theta
    self.z  = z

    return complex_dblquad(self.__RW_Bz, 0, self.alpha_max, lambda x: 0.0, lambda x: 2.0*np.pi)

  def ComputeElectromagneticField(self,r,theta,z):
    return ComputeEx(r,theta,z),ComputeEy(r,theta,z),ComputeEz(r,theta,z)

  def __RW_Ex(self,beta,alpha):
    """
    Defines the x component of the Richards-Wolf integrand.
    """
    r  = self.r
    th = self.th
    z  = self.z

    # -- We evalute some constants.
    cosAlpha  = np.cos(alpha)
    sinAlpha  = np.sin(alpha)
    cosBeta   = np.cos(beta)
    sinBeta   = np.sin(beta)
    cosBetaSq = cosBeta**2
    sinBetaSq = sinBeta**2

    # -- We obtain the value of the incident field.
    Ex_val = self.Ex(alpha,beta)
    Ey_val = self.Ey(alpha,beta)

    # -- We compute the integrand x dot E_sph.
    E_sph = self.q(alpha)*(Ex_val*(cosAlpha*cosBetaSq+sinBetaSq)+Ey_val*cosBeta*sinBeta*(cosAlpha-1))
    phase = np.exp(1j*self.k*z*cosAlpha)*np.exp(1j*self.k*r*sinAlpha*np.cos(th-beta))

    return E_sph*phase*sinAlpha

  def __RW_Ey(self,beta,alpha):
    """
    Defines the x component of the Richards-Wolf integrand.
    """
    r  = self.r
    th = self.th
    z  = self.z

    # -- We evalute some constants.
    cosAlpha  = np.cos(alpha)
    sinAlpha  = np.sin(alpha)
    cosBeta   = np.cos(beta)
    sinBeta   = np.sin(beta)
    cosBetaSq = cosBeta**2
    sinBetaSq = sinBeta**2

    # -- We obtain the value of the incident field.
    Ex = self.Ex(alpha,beta)
    Ey = self.Ey(alpha,beta)

    # -- We compute the integrand x dot E_sph.
    E_sph = self.q(alpha)*(Ex*cosBeta*sinBeta*(cosAlpha-1)+Ey*(cosAlpha*sinBetaSq+cosBetaSq))
    phase = np.exp(1j*self.k*z*cosAlpha)*np.exp(1j*self.k*r*sinAlpha*np.cos(th-beta))

    return E_sph*phase*sinAlpha

  def __RW_Ez(self,beta,alpha):
    """
    Defines the x component of the Richards-Wolf integrand.
    """
    r  = self.r
    th = self.th
    z  = self.z

    # -- We evalute some constants.
    cosAlpha  = np.cos(alpha)
    sinAlpha  = np.sin(alpha)
    cosBeta   = np.cos(beta)
    sinBeta   = np.sin(beta)
    cosBetaSq = cosBeta**2
    sinBetaSq = sinBeta**2

    # -- We obtain the value of the incident field.
    Ex = self.Ex(alpha,beta)
    Ey = self.Ey(alpha,beta)

    # -- We compute the integrand x dot E_sph.
    E_sph = self.q(alpha)*(Ex*sinAlpha*cosBeta+Ey*sinAlpha*sinBeta)
    phase = np.exp(1j*self.k*z*cosAlpha)*np.exp(1j*self.k*r*sinAlpha*np.cos(th-beta))

    return E_sph*phase*sinAlpha

  def __RW_Bx(self,beta,alpha):
    """
    Defines the x component of the Richards-Wolf integrand.
    """
    r  = self.r
    th = self.th
    z  = self.z

    # -- We evalute some constants.
    cosAlpha  = np.cos(alpha)
    sinAlpha  = np.sin(alpha)
    cosBeta   = np.cos(beta)
    sinBeta   = np.sin(beta)
    cosBetaSq = cosBeta**2
    sinBetaSq = sinBeta**2

    # -- We obtain the value of the incident field.
    Ex = self.Ex(alpha,beta)
    Ey = self.Ey(alpha,beta)

    # -- We compute the integrand x dot E_sph.
    B_sph = self.q(alpha)*(Ex*cosBeta*sinBeta*(cosAlpha-1.0)+Ey*(sinBetaSq-cosAlpha*cosBetaSq))
    phase = np.exp(1j*self.k*z*cosAlpha)*np.exp(1j*self.k*r*sinAlpha*np.cos(th-beta))

    return B_sph*phase*sinAlpha

  def __RW_By(self,beta,alpha):
    """
    Defines the x component of the Richards-Wolf integrand.
    """
    r  = self.r
    th = self.th
    z  = self.z

    # -- We evalute some constants.
    cosAlpha  = np.cos(alpha)
    sinAlpha  = np.sin(alpha)
    cosBeta   = np.cos(beta)
    sinBeta   = np.sin(beta)
    cosBetaSq = cosBeta**2
    sinBetaSq = sinBeta**2

    # -- We obtain the value of the incident field.
    Ex = self.Ex(alpha,beta)
    Ey = self.Ey(alpha,beta)

    # -- We compute the integrand x dot E_sph.
    B_sph = self.q(alpha)*(Ex*(cosAlpha*sinBetaSq+cosBetaSq)+Ey*cosBeta*sinBeta*(cosAlpha+1.0))
    phase = np.exp(1j*self.k*z*cosAlpha)*np.exp(1j*self.k*r*sinAlpha*np.cos(th-beta))

    return B_sph*phase*sinAlpha

  def __RW_Bz(self,beta,alpha):
    """
    Defines the x component of the Richards-Wolf integrand.
    """
    r  = self.r
    th = self.th
    z  = self.z

    # -- We evalute some constants.
    cosAlpha  = np.cos(alpha)
    sinAlpha  = np.sin(alpha)
    cosBeta   = np.cos(beta)
    sinBeta   = np.sin(beta)
    cosBetaSq = cosBeta**2
    sinBetaSq = sinBeta**2

    # -- We obtain the value of the incident field.
    Ex = self.Ex(alpha,beta)
    Ey = self.Ey(alpha,beta)

    # -- We compute the integrand x dot E_sph.
    B_sph = self.q(alpha)*(Ex*sinAlpha*sinBeta-Ey*sinAlpha*cosBeta)
    phase = np.exp(1j*self.k*z*cosAlpha)*np.exp(1j*self.k*r*sinAlpha*np.cos(th-beta))

    return B_sph*phase*sinAlpha

# -- We now compute the fields in the focal spot.
f    = 0.04375/UNIT_LENGTH
w0   = 0.075/UNIT_LENGTH
lamb = 800.0e-9/UNIT_LENGTH
k    = 2*np.pi/lamb

alpha_max = np.pi/2.0

def Ex(alpha,beta):
  return np.exp(-(f*np.sin(alpha)/w0)**2)

def Ey(alpha,beta):
  return 0.0

def q(alpha):
  return np.cos(alpha/2)**(-2)

hna_parabola_lin = RichardsWolf(Ex,Ey,q,alpha_max,k)

t0 = time.time()
size_r = 10
size_th= 10
r      = np.linspace(0.0,2.5e-6/UNIT_LENGTH,size_r)
th     = np.linspace(0.0,2*np.pi,size_th)
R, Th  = np.meshgrid(r,th)
X      = R*np.cos(Th)*UNIT_LENGTH
Y      = R*np.sin(Th)*UNIT_LENGTH

Ex     = np.zeros((size_r,size_th),dtype=complex)
Ey     = np.zeros((size_r,size_th),dtype=complex)
Ez     = np.zeros((size_r,size_th),dtype=complex)
Bx     = np.zeros((size_r,size_th),dtype=complex)
By     = np.zeros((size_r,size_th),dtype=complex)
Bz     = np.zeros((size_r,size_th),dtype=complex)

max_range = size_r*size_th
counter   = 1
for i in range(size_r):
  for j in range(size_th):
    Ex[i,j] = hna_parabola_lin.ComputeEx(r[i],th[j],0.0)
    Ey[i,j] = hna_parabola_lin.ComputeEy(r[i],th[j],0.0)
    Ez[i,j] = hna_parabola_lin.ComputeEz(r[i],th[j],0.0)
    Bx[i,j] = hna_parabola_lin.ComputeBx(r[i],th[j],0.0)
    By[i,j] = hna_parabola_lin.ComputeBy(r[i],th[j],0.0)
    Bz[i,j] = hna_parabola_lin.ComputeBz(r[i],th[j],0.0)

    if (counter % 100 == 0):
      print("Calculation {} of {}".format(counter,max_range))

    counter += 1

t1 = time.time()

print(t1-t0)

# -- Plot E_x for a start.
np.savetxt("x.txt", X[0,:])
np.savetxt("y.txt", Y[0,:])
np.savetxt("Ex.txt", Ex)
np.savetxt("Ey.txt", Ey)
np.savetxt("Ez.txt", Ez)
np.savetxt("Bx.txt", Bx)
np.savetxt("By.txt", By)
np.savetxt("Bz.txt", Bz)

plt.figure()
plt.pcolormesh(X/1e-6,Y/1e-6,np.transpose(np.real(Ex))/np.amax(np.real(Ex)))


# -- We now compute the fields in the axial plane.
size_z   = 100
z        = np.linspace(-1.5e-6/UNIT_LENGTH,1.5e-6/UNIT_LENGTH,size_z)
R, Z     = np.meshgrid(r,z)

Ex_axial = np.zeros((size_r,size_z),dtype=complex)
Ey_axial = np.zeros((size_r,size_z),dtype=complex)
Ez_axial = np.zeros((size_r,size_z),dtype=complex)
Bx_axial = np.zeros((size_r,size_z),dtype=complex)
By_axial = np.zeros((size_r,size_z),dtype=complex)
Bz_axial = np.zeros((size_r,size_z),dtype=complex)

max_range_axial = size_r*size_z
counter_axial  = 1
for i in range(size_r):
  for j in range(size_z):
    Ex_axial[i,j] = hna_parabola_lin.ComputeEx(r[i],0.0,z[j])
    Ey_axial[i,j] = hna_parabola_lin.ComputeEy(r[i],0.0,z[j])
    Ez_axial[i,j] = hna_parabola_lin.ComputeEz(r[i],0.0,z[j])
    Bx_axial[i,j] = hna_parabola_lin.ComputeBx(r[i],0.0,z[j])
    By_axial[i,j] = hna_parabola_lin.ComputeBy(r[i],0.0,z[j])
    Bz_axial[i,j] = hna_parabola_lin.ComputeBz(r[i],0.0,z[j])

# -- Output the fields.
np.savetxt("z.txt", z)
np.savetxt("Ex_axial.txt", Ex_axial)
np.savetxt("Ey_axial.txt", Ey_axial)
np.savetxt("Ez_axial.txt", Ez_axial)
np.savetxt("Bx_axial.txt", Bx_axial)
np.savetxt("By_axial.txt", By_axial)
np.savetxt("Bz_axial.txt", Bz_axial)
