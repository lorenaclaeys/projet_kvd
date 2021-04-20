from scipy.integrate import odeint
import math
import numpy as np
from scipy.fftpack import diff as psdiff
import matplotlib.pyplot as plt

# analytical solution
def Kdvanalytic(x,t,c,a):
    u = (c/2.)*np.cosh((math.sqrt(c)/2.)*(x - c*t - a))**(-2)
    return u

#
def eq_KdV(u,t,L):
    du_x = psdiff(u, period=L)
    dddu_x = psdiff(u, period=L,order=3)
    ut = -6*u*du_x - dddu_x
    return ut

# constants
L = 50.               # period
Nx = 200             # space step
x = np.linspace(0,L,Nx)

# parametres temporels
tmax = 200
t0 = 0
Nt = 501 #timestep
t = np.linspace(t0,tmax,Nt)

# parameters for the initial condition
c1 = 0.75
c2 = 0.4
a1 = 0.33
a2 = 0.65
u0 = Kdvanalytic(x,t0,c1,a1*L) + Kdvanalytic(x,t0,c2,a2*L)

# iintegration od the differential euqation
y = odeint(eq_KdV, u0, t, args=(L,), mxstep=5000)

#graph
plt.figure(figsize=(6,5))
plt.imshow(y[::-1, :], extent=[0,L,0,tmax])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.axis('auto')
plt.title('Korteweg-de Vries on a Periodic Domain')
plt.show()
