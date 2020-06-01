# -*- coding: utf-8 -*-


from math import *
from cmath import *
from scipy.special import *
from scipy.linalg import *
import numpy as np
import scipy as sc


#******************************************************
pi=3.1415926535897932384626433

def analiticna(T, X):

    Man=np.zeros((len(T),len(X)))

    for i in range(len(T)):
        for l in range(len(X)):

            Man[i][l]=np.sin(pi*np.cos(X[l]+T[i])) #izračunam matriko analitičnih rešitev približek

    return Man
    
def analiticna2D(dT, X):

    Man=[]

    for l in range(len(X)):

        Man.append(np.sin(pi*np.cos(X[l]+dT))) #izračunam matriko analitičnih rešitev približek

    return Man

#******************************************************

def zacetno(x):
    return np.sin(pi*np.cos(x))

#******************************************************
#from time import time # Import time-keeping library
#tstart = time()       # Define starting time
#from pylab import *   # Import libraries for plotting results


### Cash-Karp Parameters - From literature

a2,   a3,  a4,  a5,  a6      =        1/5.,    3/10.,       3/5.,            1.,        7/8.
b21, b31, b32, b41, b42, b43 =        1/5.,    3/40.,      9/40.,         3/10.,      -9/10., 6/5.
b51, b52, b53, b54           =     -11/54.,     5/2.,    -70/27.,        35/27.
b61, b62, b63, b64, b65      = 1631/55296., 175/512., 575/13824., 44275/110592.,   253/4096. 
c1,   c2,  c3,  c4,  c5, c6  =     37/378.,       0.,   250/621.,      125/594.,          0.,  512/1771.
c1star, c2star, c3star, c4star, c5star, c6star = 2825/27648., 0.,  18575/48384.,13525/55296., 277/14336., 1/4.

def stepper(deriv, k, t, y, h, tol):
   '''
   This function is called by the control function to take
   a single step forward. The inputs are the derivative function,
   the previous time and function value, the step size in time (h),
   and the tolerance for error between 5th order Runge-Kutta and 4th
   order Runge-Kutta.
   '''

   k1 = h*deriv(t,y, k)
   k2 = h*deriv(t+a2*h,y+b21*k1, k)
   k3 = h*deriv(t+a3*h,y+b31*k1+b32*k2, k)
   k4 = h*deriv(t+a4*h,y+b41*k1+b42*k2+b43*k3, k)
   k5 = h*deriv(t+a5*h,y+b51*k1+b52*k2+b53*k3+b54*k4, k)
   k6 = h*deriv(t+a6*h,y+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5, k)
   y_n_plus_1      = y +     c1*k1 +     c2*k2 +     c3*k3 +     c4*k4 +     c5*k5 +     c6*k6
   y_n_plus_1_star = y + c1star*k1 + c2star*k2 + c3star*k3 + c4star*k4 + c5star*k5 + c6star*k6
   DELTA           = y_n_plus_1 - y_n_plus_1_star
   try:
       h1 = h*abs(tol/DELTA)**0.2    # Finds step size required to meet given tolerance
   except ZeroDivisionError:
       h1 = h                        # When you are very close to ideal step, DELTA can be zero
   return t+h, y_n_plus_1, h, h1


def control(deriv, k, tol, y0, t0, h, tmax, v=False):
    '''
    This funciton takes in a python function that returns the derivative,
    a tolerance for error between RK5 and RK4, initial conditions on y (dependant) 
    and t (independant) as well as an initial step size.
    
    Keyword arguments:
    v - Verbose
    '''
#    tstart = time()
#    if v==True: print ("[%4.3f] Solving with initial condition (%0.2f, %0.2f), step size of %0.2f from t=0...%0.2f" % ((time()-tstart), y0, t0, h, tmax))

    y = [y0]      # Set up the initial conditions on y
    t = [t0]      # and t while creating the output lists
    H1= [h]   
    t_curr, y_curr, count, ncount = t0, y0, 0, 0 # Setup counters and trackers

    while t_curr < tmax:
        t_next, y_next, h, h1 = stepper(deriv, k, t_curr, y_curr, h, tol)
        if h1 < 0.9*h: 
#            if v==True: print ("[%4.3f] Reduced step size from %0.4e to %0.4e at t = %0.2f" % ((time()-tstart),h, h1, t_curr))
            h = h1
        elif h1 > 1.1*h:
#            if v==True: print ("[%4.3f] Increased step size from %0.4e to %0.4e at t = %0.2f" % ((time()-tstart),h, h1, t_curr))
            h = h1
        else:
            y.append(y_next)
            t.append(t_next)
            H1.append(h1)
            y_curr, t_curr = y_next, t_next
            ncount += 1
        count += 1
        
#    if v==True: print ("[%4.3f] Done. %i iterations, %i points" % ( (time()-tstart), count, ncount ))
    return y#, t ,H1



def Cdak0(x,ak0,k):
    dak0dx =zacetno(x)*np.exp(-1j*k*x)/(2*pi) # preveri če je to narobe
    return dak0dx

def ak0(k,dX,tol):   # tu je parameter po katerem integriramo razdalja x
    
#    y0, x0 = [0, zacetno(0)/(2*pi)], 0
    ak00, x0 = 0, 0
    h    = 0.1
#    tol  = 10**(-4) 
    Ak0= control(Cdak0,k, tol, ak00, x0, h, dX, v=True)

    L=len(Ak0)-1
    
    return Ak0[L]


def Cdak(t,ak,k):
    dakdt =  1j*k*ak
    return dakdt

def ak(k,dT,dX,tol):  # tu je parameter po katerem integriramo čas t

    Ak0, t0 =ak0(k,dX,tol),0 # začetna pogoja izračunamo
    h    = 0.1
#    tol  = 10**(-4) 
    Ak= control(Cdak, k, tol, Ak0, t0, h, dT, v=True)
    
    L=len(Ak)-1
        
    return Ak[L]


# analitična rešitev za koeficiente ak(t)
def akAn(k,t):
    return np.sin(k*pi/2)*sc.special.jn(k,pi)*np.exp(1j*k*t)
    

# izračun hitrosti 3D
def u3D(a,b,delitev,dt,dT,n,tol):

    dx = (b-a)/delitev

    U=np.zeros((int(dT/dt)+1,delitev+1))

    X = np.linspace(a, b, delitev+1)
    T = np.linspace(0, dT, int(dT/dt)+1)

    for i in range(int(dT/dt)+1):
        t = dt*i
        print(t)
        for j in range(delitev+1):
            x = dx*j

#            U[i][j]=sum(ak(k,t,b) * np.exp(1j*k*x) for k in range(-int(n/2),int(n/2)+1,1))
            U[i][j]=sum(akAn(k,t) * np.exp(1j*k*x) for k in range(-int(n/2),int(n/2)+1,1))

    return X,T,U.real



## izračun hitrosti 2D

def u2D(a,b,delitev,dT,n,tol,nacin):

    dx = (b-a)/delitev

    U=[]

    for j in range(delitev+1):        
        x = dx*j
        if nacin == 1 :
            U.append(sum(ak(k,dT,b,tol) * np.exp(1j*k*x) for k in range(-int(n/2),int(n/2)+1,1)))
        else:
            U.append(sum(akAn(k,dT) * np.exp(1j*k*x) for k in range(-int(n/2),int(n/2)+1,1)))
            
    U=np.array(U)

    return U.real
    
#*********************** IZRAČUNI 3D galerkin ****************************

a=0
b=2*pi  # dolžina krajevnega intervala
N=100 # delitev krajevnega intervala
#dx=(b-a)/Nx # za N=100 0.06283185307179587

dT=20*pi# dolžina časovnega intervala

##dt=0.1  # za N=300 --> 0.14
dt=0.005  # za N=300 --> 0.14
#dt=0.0005  # za N=300 --> 0.14

n=10 # parameter dolžine vsote - grafa 3D in 2D
#n=100 # parameter dolžine vsote - graf 2D

k=n/2
tol  = 10**(-4)
AK4=ak(k,dT,b,tol)
print(AK4)
tol  = 10**(-10)  # čez noč
AK10=ak(k,dT,b,tol)
print(AK10)
AKnm=akAn(k,dT)
print(AKnm)

#AA=ak(1,2*pi,b)
#BB=akAn(1,2*pi)
#
#print(AA)
#print(BB)

#X,T,M = u3D(a,b,N,dt,dT,n,tol)
#
#Man = analiticna(T,X)

##********************** GRAFIRANJE 3D ***************************

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

from matplotlib import colors, ticker, cm

import matplotlib.ticker as mticker
def log_tick_formatter(val, pos=None):
    return "{:.0e}".format(10**val)

#**********************
# nalogi a.) in b.)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#XX,TT = np.meshgrid(X,T)
#
##surf = ax.plot_wireframe(TT,XX,M, color = 'y', rstride=10, cstride=0, label='diferenčna')
#surf = ax.plot_surface(TT,XX,M, rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0, label='Galerkin')
##surf = ax.plot_wireframe(TT,XX,Man, color = 'k', rstride=10, cstride=0, label='analitična')
##surf = ax.plot_surface(TT,XX,Man, rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0, label='analitična')
#
##surf = ax.plot_surface(TT,XX,abs(Man-M), rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0, label='odstopanje')
##
##surf = ax.plot_surface(TT,XX,np.log10(abs(Man-M)), rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0, label='odstopanje')
##ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
#
##surf = ax.plot_surface(TT[0:150,:],XX[0:150,:],M[0:150,:], color = 'y', rstride=10, cstride=0, label='Galerkin')
##surf = ax.plot_wireframe(TT[0:150,:],XX[0:150,:],Man[0:150,:], color = 'k', rstride=10, cstride=0, label='analitična')
##ax.set_title( 'Odstopanje rešitve hiperbolične valovne enačbe po Galerkinu od analitične (Galerkin $N_{x}=$'+str(N)+', $dt=$'+str(dt)+', $N=$'+str(n)+', $\epsilon_{RK4}=$'+str(tol)+')')
#ax.set_title( 'Časovni razvoj za hiperbolično valovno enačbo (GalerkinAN $N_{x}=$'+str(N)+', $dt=$'+str(dt)+', $N=$'+str(n)+', $\epsilon_{RK4}=$'+str(tol)+')')
##ax.set_title( 'Časovni razvoj za hiperbolično valovno enačbo (analitična rešitev)')
#ax.set_xlabel("t")
#ax.set_ylabel("x")
#ax.set_zlabel("$u(x,t)$")
##ax.set_zlabel("$|u_{Gal}-u_{an}|$")
##ax.set_zlabel("$log_{10}|u_{Gal}-u_{an}|$")
#ax.set_ylim([0,2*pi])
##ax.legend(loc='best')
#plt.tight_layout()
#plt.show()



##*********************** IZRAČUNI 3D galerkin ****************************
#
#
#dT=20*pi
#
#U10_4=u2D(a,b,50,dT,10,10**(-4),1)
#print('U10_4')
#U10_10=u2D(a,b,50,dT,10,10**(-10),1)
#print('U10_10')
#U10_an=u2D(a,b,50,dT,10,tol,2)
#print('U10_an')
#
#
#U100_4=u2D(a,b,50,dT,20,10**(-4),1)
#print('U100_4')
#U100_10=u2D(a,b,50,dT,20,10**(-10),1)
#print('U100_10')
#U100_an=u2D(a,b,50,dT,20,tol,2)
#print('U100_an')
#
#X = np.linspace(a, b, 50+1)
#Uan=analiticna2D(dT,X)
##
##
###********************** GRAFIRANJE 3D ***************************
#


## graf 2D hitrosti ob času dT
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#
#ax2.plot(X,U10_4,"y:",label='Galerkin - $\epsilon=10^{-4}$, $N=10$') ## različni časovni koraki
#ax2.plot(X,U10_10,"y-.",label='Galerkin - $\epsilon=10^{-10}$, $N=10$') ## različni časovni koraki
#ax2.plot(X,U10_an,"y--",label='GalerkinAN - $N=10$')
#
#ax2.plot(X,U100_4,"g:",label='Galerkin - $\epsilon=10^{-4}$, $N=20$') ## različni časovni koraki
#ax2.plot(X,U100_10,"g-.",label='Galerkin - $\epsilon=10^{-10}$, $N=20$') ## različni časovni koraki
#ax2.plot(X,U100_an,"g--",label='GalerkinAN - $N=20$')
#
#ax2.plot(X,Uan,"k:",label='analitična')
#
#ax2.legend(loc='best')
#ax2.set_title( 'Rešitve hiperbolične valovne enačbe ob času $t=20\pi$')
#plt.xlabel('x')
#plt.ylabel("$u(x,t)$")
#plt.xlim([0,2*pi])
#plt.show()
#
#
## graf 2D odstopanja hitrosti ob času dT
#fig3 = plt.figure()
#ax3 = fig3.add_subplot(111)
#
#ax3.plot(X,abs(U10_4-Uan),"y:",label='Galerkin - $\epsilon=10^{-4}$, $N=10$') ## različni časovni koraki
#ax3.plot(X,abs(U10_10-Uan),"y-.",label='Galerkin - $\epsilon=10^{-10}$, $N=10$') ## različni časovni koraki
#ax3.plot(X,abs(U10_an-Uan),"y--",label='GalerkinAN - $N=10$')
#
#ax3.plot(X,abs(U100_4-Uan),"g:",label='Galerkin - $\epsilon=10^{-4}$, $N=20$') ## različni časovni koraki
#ax3.plot(X,abs(U100_10-Uan),"g-.",label='Galerkin - $\epsilon=10^{-10}$, $N=20$') ## različni časovni koraki
#ax3.plot(X,abs(U100_an-Uan),"g--",label='GalerkinAN - $N=20$')
#
#ax3.legend(loc='best')
#ax3.set_title( 'Odstopanja od točne rešitve hiperbolične valovne enačbe ob času $t=20\pi$')
#plt.xlabel('x')
#plt.ylabel("$|u_{Gal}-u_{an}|$")
#plt.xlim([0,2*pi])
#plt.yscale('log')
#plt.show()


