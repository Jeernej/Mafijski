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

#******************************************************

def zacetno(x):
    return np.sin(pi*np.cos(x))

#******************************************************
from scipy.integrate import ode


#
#def f(t, y, arg1):
#    return [1j*arg1*y[0] + y[1], -arg1*y[1]**2]
#def jac(t, y, arg1):
#    return [[1j*arg1, 1], [0, -arg1*2*y[1]]]
#
#y0, t0 = [1.0j, 2.0], 0
#r = ode(f).set_integrator('zvode', method='bdf')
#r.set_initial_value(y0, t0).set_f_params(2.0)#.set_jac_params(2.0)
#t1 = 10
#dt = 10
#eps=10.**(-11)    
#while r.successful() and r.t < t1*eps:
#    print(r.t+dt, r.integrate(r.t+dt))


# izračun ak(0)
#def dak0(y,x,k,l):
#    ak0, dak0 = y
#    dydt = [dak0, (l*zacetno(x)*np.exp(-1j*k*x)/(2*pi)).real] # preveri če je to narobe
#    return dydt

def Cdak0(x,y,k):
    ak0, dak0 = y
#    dydt = [dak0, (zacetno(x)*np.exp(-1j*k*x)/(2*pi))] # preveri če je to narobe
    dydt = [dak0,  -np.exp(-1j*k*x)*(1j*k*np.sin(pi*np.cos(x)) + pi*np.sin(x) *np.cos(pi* np.cos(x)))] # preveri če je to narobe
    return dydt

def ak0(k,dX):   # tu je parameter po katerem integriramo razdalja x
#    l=1
#    sol = sc.integrate.complex_ode(Cdak0(X,[0, zacetno(0)/(2*pi)],k,l)) #pribl = [ak00, dak00] začetna pogoja
#    sol = sc.integrate.odeint(dak0, [0, zacetno(0)/(2*pi)],X,args=(k,l)) #pribl = [ak00, dak00] začetna pogoja
#    sol = sc.integrate.odeint(dak0, [akAn(k,0), zacetno(0)/(2*pi)],X,args=(k)) #pribl = [ak00, dak00] začetna pogoja
    
#    y0, x0 = [0, zacetno(0)/(2*pi)], 0
    y0, x0 = [0, 0], 0
    
    r = ode(Cdak0).set_integrator('zvode', method='bdf')
    r.set_initial_value(y0, x0).set_f_params(k)#.set_jac_params(2.0)

#    Lx=len(X)-1    
#    dx = X[Lx]/Lx
#    sol=[]
    eps=10.**(-11)    
    while r.successful() and r.t < dX+eps: # AttributeError: 'ode' object has no attribute 'x' - zato moram uporabit t
#        sol.append(r.integrate(r.t+dx)[0])    # pripenjam le rešitve ak0 od rešitve array([ak0,dak0])
        Ak0=(r.integrate(r.t+dX)[0])    # končna rešitev ak0 od rešitve array([ak0,dak0])
    
#    Ak0=sol
#    dAk0=sol[:,1]
#    L=len(Ak0)-1
    
    return Ak0

# izračun ak(t)

#def dak(y,t,k,l):
#    ak, dak = y
#    dydt = [dak, (l*1j*k*ak).real]  # preveri če je to narobe
#    return dydt

def Cdak(t,y,k):
    ak, dak = y
#    dydt = [dak, (1j*k*ak)]  # preveri če je to narobe
    dydt = [dak, 1j*k*dak]  # preveri če je to narobe
    return dydt

def ak(k,dT,dX):  # tu je parameter po katerem integriramo čas t
#    l=1
    Ak0=ak0(k,dX) # začetna pogoja izračunamo
#    sol = sc.integrate.complex_ode(Cdak(T,[Ak0, dAk0],k,l)) #pribl = [ak0, dak0]  začetna pogoja
#    sol = sc.integrate.odeint(dak, [Ak0, dAk0],T,args=(k,l)) #pribl = [ak0, dak0]  začetna pogoja
    
#    y0, t0 = [Ak0, zacetno(0)/(2*pi)], 0
    y0, t0 = [0, 1j*k*Ak0], 0
    
    r = ode(Cdak).set_integrator('zvode', method='bdf')#, rtol=1e-9, atol=1e-12)
    r.set_initial_value(y0, t0).set_f_params(k)#.set_jac_params(2.0)

#    Lt=len(T)-1    
#    dt = T[Lt]/Lt
#    sol=[]
    eps=10.**(-11)
    while r.successful() and r.t < dT+eps:
#        sol.append(r.integrate(r.t+dx)[0])    # pripenjam le rešitve ak0 od rešitve array([ak0,dak0])
        Ak=(r.integrate(r.t+dT)[0])
        
#    Ak=sol[:,0]
##    dAk=sol[:,1]
#    L=len(Ak)-1
        
    return Ak


# analitična rešitev za koeficiente ak(t)
def akAn(k,t):
    return np.sin(k*pi/2)*sc.special.jn(k,pi)*np.exp(1j*k*t)

AA=ak(1,pi/3,b)
BB=akAn(1,pi/3)

print(AA)
print(BB)


# izračun hitrosti
def u(a,b,delitev,dt,dT,n):

    dx = (b-a)/delitev

    U=np.zeros((int(dT/dt)+1,delitev+1))

    X = np.linspace(a, b, delitev+1)
    T = np.linspace(0, dT, int(dT/dt)+1)

    for i in range(int(dT/dt)+1):
        t = dt*i
        print(t)
        for j in range(delitev+1):
            x = dx*j

            U[i][j]=sum(ak(k,t,T,X) * np.exp(1j*k*x) for k in range(-int(n/2),int(n/2)+1,1))
#            U[i][j]=sum(akAn(k,t) * np.exp(1j*k*x) for k in range(-int(n/2),int(n/2)+1,1))

    return X,T,U.real


#*********************** IZRAČUNI galerkin ****************************

a=0
b=2*pi  # dolžina krajevnega intervala
N=50 # delitev krajevnega intervala
#dx=(b-a)/Nx # za N=100 0.06283185307179587

dT=4*pi# dolžina časovnega intervala

##dt=0.1  # za N=300 --> 0.14
dt=0.5  # za N=300 --> 0.14
#dt=0.0005  # za N=300 --> 0.14

n=20 # parameter dolžine vsote

#X,T,M = u(a,b,N,dt,dT,n)
#
#Man = analiticna(T,X)

##********************** GRAFIRANJE ***************************

#
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d
#from matplotlib import cm
#
#from matplotlib import colors, ticker, cm
#
#import matplotlib.ticker as mticker
#def log_tick_formatter(val, pos=None):
#    return "{:.0e}".format(10**val)
#
##**********************
### nalogi a.) in b.)
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
##ax.set_title( 'Odstopanje rešitve hiperbolične valovne enačbe po Galerkinu od analitične (Galerkin $N_{x}=$'+str(N)+', $dt=$'+str(dt)+', $N=$'+str(n)+')')
#ax.set_title( 'Časovni razvoj za hiperbolično valovno enačbo (Galerkin $N_{x}=$'+str(N)+', $dt=$'+str(dt)+', $N=$'+str(n)+')')
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
#


#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#for i in range(int(5)):
#    ax2.plot(X,abs(M[1+i*int(len(T)/5),:]-Man[1+i*int(len(T)/5),:]),"r:")
#
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"b:",label='$dt=0.01\cdot dx^2$') ## različni časovni koraki
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"g:",label='$dt=0.1dx^2$')
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"y:",label='$dt=2\cdot dx^2$')
#ax2.plot(X,abs(M[len(T)-1,:]-Man[len(T)-1,:]),"r:",label='$dt=8\cdot dx^2$')
#
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"b:",label='$N=1000$') ## različn delitve krajevnega intervala
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"g:",label='$N=500$')
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"y:",label='$N=300$')
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"r:",label='$N=100$')
#
#ax2.legend(loc='lower right')
#ax2.set_title( 'Odstopanja od točne rešitve hiperbolične valovne enačbe pri razvoju skozi čas')
#plt.xlabel('x')
#plt.ylabel("$u(x,t)$")
#plt.xlim([0,2*pi])
#plt.yscale('log')
#plt.show()
