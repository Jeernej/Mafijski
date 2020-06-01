# -*- coding: utf-8 -*-
#from math import *
import scipy as sc
import numpy  as np
from numpy import linalg
from cmath import *

# *************************************************
pi=3.1415926535897932384626433

##square root of -1: Python adheres to this convention: a number followed by “j” is treated as an imaginary number!! i->1j

#***********************************************************************************************************************
## a) RAZVOJ ZAČETNEGA STANJA V PROSTORU S POTENCIALOM

def V(x,k):
    return 0.5*k*x**2

def zacetno(x, alfa, lam ):
    return np.sqrt(alfa/np.sqrt(pi))*sc.exp(-0.5*alfa**2*(x-lam)**2)

def analiticna(t, x, alfa, lam, w):

    Man=np.zeros((len(T),len(X)))

    for i in range(len(T)):
        for l in range(len(X)):
            koef = np.sqrt(alfa/np.sqrt(pi))
            clen1 = -0.5*(alfa*X[l]-alfa*lam*np.cos(w*T[i]))**2
            clen2 = -1j*(w*T[i]/2 + lam*X[l]*alfa**2*np.sin(w*T[i]) - 0.25*np.sin(2*w*T[i])*(alfa*lam)**2)
            Man[i][l]=abs(koef*sc.exp(clen1+clen2)) #izračunam matriko analitičnih rešitev približek

    return Man

def dif2(dt,dx,N,x0,k):
    A = np.zeros((N+1,N+1),np.complex_)
    b = 1j*dt/(2*dx**2)
    a = -b/2.0
    for i in range(N):
        A[i][i+1],A[i+1][i] = a, a
    for i in range(N+1):
        A[i][i] = 1 + b + 1j*dt/2.0*V(x0+i*dx,k)
    Az = A.conj()
    return A,Az

## dodatna
def dif4(dt,dx,N,x0,k):

    A = np.zeros((N+1,N+1),np.complex_)
    b = dt/(2*dx**2)*1j

    ee = -4./6*b
    ff = 1./24*b

    for i in range(N-1):
        A[i][i+2],A[i+2][i] = ff, ff
    for i in range(N):
        A[i][i+1],A[i+1][i] = ee, ee
    for i in range(N+1):
        A[i][i] = 1 + 5./4*b + 1j*dt/2.0*V(x0+i*dx,k)

    Acon = A.conjugate()

    return A, Acon

def dif6(dt,dx,N,x0,k):

    A = np.zeros((N+1,N+1),np.complex_)
    b = dt/(2*dx**2)*1j

    ee = -3.0/4*b
    ff = 3.0/40*b
    gg = -1.0/180*b

    for i in range(N-2):
        A[i][i+3],A[i+3][i] = gg, gg
    for i in range(N-1):
        A[i][i+2],A[i+2][i] = ff, ff
    for i in range(N):
        A[i][i+1],A[i+1][i] = ee, ee
    for i in range(N+1):
        A[i][i] = 1 + 49./36*b + 1j*dt/2.0*V(x0+i*dx,k)

    Acon = A.conjugate()

    return A, Acon

def dif8(dt,dx,N,x0,k):

    A = np.zeros((N+1,N+1),np.complex_)
    b = dt/(2*dx**2)*1j

    ee = -8./10*b
    ff = 1./10*b
    gg = -8./630*b
    hh = 1./1120*b

    for i in range(N-3):
        A[i][i+4],A[i+4][i] = hh, hh
    for i in range(N-2):
        A[i][i+3],A[i+3][i] = gg, gg
    for i in range(N-1):
        A[i][i+2],A[i+2][i] = ff, ff
    for i in range(N):
        A[i][i+1],A[i+1][i] = ee, ee
    for i in range(N+1):
        A[i][i] = 1 + 205./144*b + 1j*dt/2.0*V(x0+i*dx,k)

    Acon = A.conjugate()

    return A, Acon

def razvoj(dx,dt,N,a, b, k0,x0,dT,alfa,lam, diff):

#    dx=(b-a)/N ## koraki na krajevnem intervalu
    X=[]
#    dt=6*dx**2 ## koraki na casovnem intervalu naredi kompromis
    T=[]
    T.append(0)

    psi0=[]
    for i in range(N+1) :
        psi0.append(zacetno(x0+i*dx, alfa, lam )) #izračunam začetni približek
        X.append(x0+i*dx) #izračunam točke krajevnega intervala

    Mg=psi0  # v matriko časovnih korakov vpišem začetni približek ob času t=0

    if diff==2:
        A,Acon = dif2(dt,dx,N,x0, k0)
    elif diff==4:
        A,Acon = dif4(dt,dx,N,x0, k0)
    elif diff==6:
        A,Acon = dif6(dt,dx,N,x0, k0)
    elif diff==8:
        A,Acon = dif8(dt,dx,N,x0, k0)

    cas=1
    k=0. #pogoj za vstop v zanko
    while dt*k<10*dT:

        psi1 = np.linalg.solve(A,np.dot(Acon,psi0)) #rešim sistem za vrednost v novem časovnem koraku dt0

        if cas==1: Mg=np.concatenate(([Mg],[psi1]), axis=0) # v matriko časovnih korakov vpišem vrednost v novem časovnem koraku dt0
        else: Mg=np.concatenate((Mg,[psi1]), axis=0)

        psi0=psi1
        T.append(cas*dt) #računam točke časovnega intervala
        cas=cas+1
        k=k+1

    return T, X, Mg


#***********************************************************************************************************************
# b)   RAZVOJ GAUSSOVEGA VALOVNEGA PAKETA V PROSTORU BREZ POTENCIALA

def analiticnaGauss(T,X, sig, k, lamb):

    Man=np.zeros((len(T),len(X)))

    for i in range(len(T)):
        for l in range(len(X)):
            koef = (2*pi*sig**2)**(-0.25)/sc.sqrt(1 + 1j*T[i]/(2*sig**2))
            stevec = -((X[l]-lamb)/(2*sig))**2 + 1j*k*(X[l]-lamb) - 1j*T[i]*k**2/2
            Man[i][l]=abs(koef*sc.exp(stevec/(1 + 1j*T[i]/(2*sig**2)))) #izračunam matriko analitičnih rešitev približek

    return Man

def zacetnoGauss(x, sig, k, lamb):

    koef = (2*pi*sig**2)**(-0.25)
    clen1 = sc.exp(1j*k*(x-lamb))
    clen2 = sc.exp(-((x-lamb)/(2*sig))**2)

    return koef*clen1*clen2

def difGauss2(dt,dx,N):

    d = 1.+1j*dt/(2*dx**2.)  # diagonalni za časovni korak dt0 in krajevni korak dx0
    a = -1j*dt/(4.*dx**2.)  # obdiagonalni za časovni korak dt0 in krajevni korak dx0

    A = d*np.identity(N+1)    # ustvarimo matriko A
    for i in range(N):
        A[i][i+1],A[i+1][i] = a, a   # ustvarimo matriko A
    Acon = A.conjugate()

    return A, Acon

### dodatna
def difGauss4(dt,dx,N):

    A = np.zeros((N+1,N+1),np.complex_)
    b = dt/(2*dx**2)*1j

    ee = -4./6*b
    ff = 1./24*b

    for i in range(N-1):
        A[i][i+2],A[i+2][i] = ff, ff
    for i in range(N):
        A[i][i+1],A[i+1][i] = ee, ee
    for i in range(N+1):
        A[i][i] = 1 + 5./4*b

    Acon = A.conjugate()

    return A, Acon

def difGauss6(dt,dx,N):

    A = np.zeros((N+1,N+1),np.complex_)
    b = dt/(2*dx**2)*1j

    ee = -3.0/4*b
    ff = 3.0/40*b
    gg = -1.0/180*b

    for i in range(N-2):
        A[i][i+3],A[i+3][i] = gg, gg
    for i in range(N-1):
        A[i][i+2],A[i+2][i] = ff, ff
    for i in range(N):
        A[i][i+1],A[i+1][i] = ee, ee
    for i in range(N+1):
        A[i][i] = 1 + 49./36*b

    Acon = A.conjugate()

    return A, Acon

def difGauss8(dt,dx,N):

    A = np.zeros((N+1,N+1),np.complex_)
    b = dt/(2*dx**2)*1j

    ee = -8./10*b
    ff = 1./10*b
    gg = -8./630*b
    hh = 1./1120*b

    for i in range(N-3):
        A[i][i+4],A[i+4][i] = hh, hh
    for i in range(N-2):
        A[i][i+3],A[i+3][i] = gg, gg
    for i in range(N-1):
        A[i][i+2],A[i+2][i] = ff, ff
    for i in range(N):
        A[i][i+1],A[i+1][i] = ee, ee
    for i in range(N+1):
        A[i][i] = 1 + 205./144*b

    Acon = A.conjugate()

    return A, Acon



def razvojGauss(dx,dt,N,a0, b0, k0, sig0, lamb, diff):

#    dx=(b0-a0)/N ## koraki na krajevnem intervalu dx€[0,0.75]
    X=[]
#    dt=2*dx**2 ## koraki na casovnem intervalu
    T=[]
    T.append(0)

    psi0=[]
    for i in range(N+1) :
        psi0.append(zacetnoGauss(a0+i*dx, sig0, k0, lamb)) #izračunam začetni približek
        X.append(a0+i*dx) #izračunam točke krajevnega intervala

    Mg=psi0  # v matriko časovnih korakov vpišem začetni približek ob času t=0

    if diff==2:
        A,Acon = difGauss2(dt,dx,N)
    elif diff==4:
        A,Acon = difGauss4(dt,dx,N)
    elif diff==6:
        A,Acon = difGauss6(dt,dx,N)
    elif diff==8:
        A,Acon = difGauss8(dt,dx,N)

    cas=1
    tez=0. #pogoj za vstop v zanko
    while tez < 0.75:

        psi1 = np.linalg.solve(A,np.dot(Acon,psi0)) #rešim sistem za vrednost v novem časovnem koraku dt0

        if cas==1: Mg=np.concatenate(([Mg],[psi1]), axis=0) # v matriko časovnih korakov vpišem vrednost v novem časovnem koraku dt0
        else: Mg=np.concatenate((Mg,[psi1]), axis=0)

        tez = sum([dx*(a0+(i + 0.5)*dx)*(abs(psi0[i])**2+abs(psi0[i + 1])**2)/2. for i in range(N-1)])

        psi0=psi1
        T.append(cas*dt) #računam točke časovnega intervala
        cas=cas+1

    return T, X, Mg

#*********************** IZRAČUNI ****************************

#N=50

#N=100
N=300
#N=500

#N=1000

# ****************** paket v potencialu  *******************************

k = 0.04
lam = 10.
w = np.sqrt(k)
alfa = k**0.25
dT=2.*pi/w

a=-40.
b=40.
x0=-40.
dx=(b-a)/N

#dt=0.14
#dt=0.1*dx**2 # za N=300 --> 0.007
#dt=dx**2  # za N=300 --> 0.07
dt=2*dx**2  # za N=300 --> 0.14
#dt=4*dx**2 # za N=300 --> 0.285
#dt=8*dx**2 # za N=300 --> 0.57

T, X, M = razvoj(dx,dt,N,a, b, k, x0,dT,alfa,lam, 2)
#dodatna
T4, X4, M4 = razvoj(dx,dt,N,a, b, k, x0,dT,alfa,lam,4)
T6, X6, M6 = razvoj(dx,dt,N,a, b, k, x0,dT,alfa,lam,6)
T8, X8, M8 = razvoj(dx,dt,N,a, b, k, x0,dT,alfa,lam,8)

Man = analiticna(T,X, alfa, lam, w)

# ****************** Gausovski paket  brez potenciala  *******************************

#k0 = 50*pi
#sig0 = 1./20
#lamb = 0.25
#
#a0=-0.5
#b0=1.5
#dx=(b0-a0)/N
#
##dt=8.88888888888889*10**(-5)
##dt=0.01*dx**2 # za N=300 -->
##dt=0.1*dx**2 # za N=300 -->
#dt=2*dx**2  # za N=300 --> 8.88888888888889e-05
##dt=8*dx**2 # za N=300 --> 110.44661672776617
#
#T, X, M = razvojGauss(dx,dt,N,a0,b0,k0,sig0,lamb,2)
##dodatna
#T4, X4, M4 = razvojGauss(dx,dt,N,a0,b0,k0,sig0,lamb,4)
#T6, X6, M6 = razvojGauss(dx,dt,N,a0,b0,k0,sig0,lamb,6)
#T8, X8, M8 = razvojGauss(dx,dt,N,a0,b0,k0,sig0,lamb,8)
#
#Man = analiticnaGauss(T,X, sig0, k0, lamb)

#********************** GRAFIRANJE ***************************

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

#**********************
## nalogi a.) in b.)
fig = plt.figure()
ax = fig.gca(projection='3d')
XX,TT = np.meshgrid(X,T)
XX4,TT4 = np.meshgrid(X4,T4)
XX6,TT6 = np.meshgrid(X6,T6)
XX8,TT8 = np.meshgrid(X8,T8)

#surf = ax.plot_wireframe(TT,XX,abs(M)**2, color = 'y', rstride=10, cstride=0, label='n=2')
#surf = ax.plot_wireframe(TT4,XX4,abs(M4)**2, color = 'blue', rstride=10, cstride=0, label='n=4')
#surf = ax.plot_wireframe(TT6,XX6,abs(M6)**2, color = 'green', rstride=10, cstride=0,label='n=6')
#surf = ax.plot_wireframe(TT8,XX8,abs(M8)**2, color = 'red', rstride=10, cstride=0, label='n=8')
#surf = ax.plot_wireframe(TT,XX,abs(Man)**2, color = 'k', rstride=10, cstride=0, label='analitična')

surf = ax.plot_wireframe(TT[0:150,:],XX[0:150,:],abs(M[0:150,:])**2, color = 'y', rstride=10, cstride=0, label='n=2')
surf = ax.plot_wireframe(TT4[0:150,:],XX4[0:150,:],abs(M4[0:150,:])**2, color = 'blue', rstride=10, cstride=0, label='n=4')
surf = ax.plot_wireframe(TT6[0:150,:],XX6[0:150,:],abs(M6[0:150,:])**2, color = 'green', rstride=10, cstride=0,label='n=6')
surf = ax.plot_wireframe(TT8[0:150,:],XX8[0:150,:],abs(M8[0:150,:])**2, color = 'red', rstride=10, cstride=0, label='n=8')
surf = ax.plot_wireframe(TT[0:150,:],XX[0:150,:],abs(Man[0:150,:])**2, color = 'k', rstride=10, cstride=0, label='analitična')


#ax.plot_surface(TT, XX, M, rstride=8, cstride=8, alpha=0.3)
#cset = ax.contourf(TT, XX, M, zdir='m', offset=-100, cmap=cm.coolwarm)
#cset = ax.contourf(TT, XX, M, zdir='t', offset=-40, cmap=cm.coolwarm)
#cset = ax.contourf(TT, XX, M, zdir='y', offset=40, cmap=cm.coolwarm)
ax.set_title( 'Časovni razvoj začetnega kvantnega stanja')
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_zlabel("$|\psi(x,t)|^2$")
#ax.set_ylim([-20,20])
ax.legend(loc='best')
plt.tight_layout()
plt.show()


## naloga a.)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
#for i in range(int(7)):
#    ax1.plot(X,abs(M[i*int(dT/dt/2/7),:])**2,"y--")#,label='t=0.001')
#    ax1.plot(X4,abs(M4[i*int(dT/dt/2/7),:])**2,"y--")#,label='t=0.001')
#    ax1.plot(X6,abs(M6[i*int(dT/dt/2/7),:])**2,"y--")#,label='t=0.001')
#    ax1.plot(X8,abs(M8[i*int(dT/dt/2/7),:])**2,"y--")#,label='t=0.001')
#    ax1.plot(X,abs(Man[i*int(dT/dt/2/7),:])**2,"k--")#,label='t=0.001')
#    ax1.plot(X,abs(M[2209-(i+1)*int(dT/dt/2/7),:])**2,"y:")#,label='t=0.001')
#    ax1.plot(X4,abs(M4[2209-(i+1)*int(dT/dt/2/7),:])**2,"y:")#,label='t=0.001')
#    ax1.plot(X6,abs(M6[2209-(i+1)*int(dT/dt/2/7),:])**2,"y:")#,label='t=0.001')
#    ax1.plot(X8,abs(M8[2209-(i+1)*int(dT/dt/2/7),:])**2,"y:")#,label='t=0.001')
#    ax1.plot(X,abs(Man[2209-(i+1)*int(dT/dt/2/7),:])**2,"k:")#,label='t=0.001')

ax1.plot(X,abs(M[8*int(dT/dt/2/7),:])**2,"y--",label='n=2; (0<t<T/2)')
ax1.plot(X4,abs(M4[8*int(dT/dt/2/7),:])**2,"b--",label='n=4; (0<t<T/2)')
ax1.plot(X6,abs(M6[8*int(dT/dt/2/7),:])**2,"g--",label='n=6; (0<t<T/2)')
ax1.plot(X8,abs(M8[8*int(dT/dt/2/7),:])**2,"r--",label='n=8; (0<t<T/2)')
ax1.plot(X,abs(Man[8*int(dT/dt/2/7),:])**2,"k--",label='analiticna; (0<t<T/2)')

ax1.plot(X,abs(M[2209,:])**2,"y:",label='n=2; (19T/2<t<10T)')
ax1.plot(X,abs(M[2209,:])**2,"b:",label='n=4; (19T/2<t<10T)')
ax1.plot(X,abs(M[2209,:])**2,"g:",label='n=6; (19T/2<t<10T)')
ax1.plot(X,abs(M[2209,:])**2,"r:",label='n=8; (19T/2<t<10T)')
ax1.plot(X,abs(Man[2209,:])**2,"k:",label='analiticna; (19T/2<t<10T)')

#ax1.plot(X,M[0,:],"k",label='t=0.001')
#ax1.plot(X,M[100,:],"k--",label='t=0.1')
#ax1.plot(X,M[100000,:],"k-.",label='t=100')
#ax1.plot(X,M[200000,:],"k:",label='t=200')
#ax1.plot(X,M[400000,:],"k:",label='t=200')
ax1.legend(loc='best')
ax1.set_title( 'Časovni razvoj začetnega kvantnega stanja')
plt.xlabel('x')
plt.ylabel('$|\psi(x,t)|^2$')
plt.xlim([-20,20])
plt.show()



#
##
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#for i in range(int(8)):
#    ax2.plot(X,abs(abs(M[int(len(T)/10/4)+i*int(len(T)/10),:])**2-abs(Man[int(len(T)/10/4)+i*int(len(T)/10),:])**2),"k:")#,label='t=0.001')
#
##ax2.plot(X,abs(abs(Man[int(len(T)-len(T)/10/4),:])**2-abs(Man[int(len(T)-len(T)/10/4),:])**2),"b:",label='$dt=0.1\cdot dx^2$')
##ax2.plot(X,abs(abs(Man[int(len(T)-len(T)/10/4),:])**2-abs(Man[int(len(T)-len(T)/10/4),:])**2),"g:",label='$dt=dx^2$')
##ax2.plot(X,abs(abs(Man[int(len(T)-len(T)/10/4),:])**2-abs(Man[int(len(T)-len(T)/10/4),:])**2),"y:",label='$dt=2\cdot dx^2$')
##ax2.plot(X,abs(abs(Man[int(len(T)-len(T)/10/4),:])**2-abs(Man[int(len(T)-len(T)/10/4),:])**2),"r:",label='$dt=4\cdot dx^2$')
##ax2.plot(X,abs(abs(Man[int(len(T)-len(T)/10/4),:])**2-abs(Man[int(len(T)-len(T)/10/4),:])**2),"k:",label='$dt=8\cdot dx^2$')
#
##ax2.plot(X,abs(abs(Man[(len(T)-len(T)/10/4),:])**2-abs(Man[(len(T)-len(T)/10/4),:])**2),"g:",label='$N=500$')
##ax2.plot(X,abs(abs(Man[(len(T)-len(T)/10/4),:])**2-abs(Man[(len(T)-len(T)/10/4),:])**2),"y:",label='$N=300$')
##ax2.plot(X,abs(abs(Man[(len(T)-len(T)/10/4),:])**2-abs(Man[(len(T)-len(T)/10/4),:])**2),"r:",label='$N=100$')
#ax2.plot(X,abs(abs(Man[(len(T)-len(T)/10/4),:])**2-abs(Man[(len(T)-len(T)/10/4),:])**2),"k:",label='$N=50$')
#
#ax2.legend(loc='lower right')
#ax2.set_title( 'Odstopanja od točne rešitve pri razvoju začetnega kvantnega stanja skozi čas (n=2)')
#plt.xlabel('x')
#plt.ylabel('$||\psi_{an}|^2-|\psi_{nu}|^2|$')
#plt.xlim([-15,15])
#plt.yscale('log')
#plt.show()

## naloga b.)
#
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#for i in range(int(5)):
#    ax2.plot(X,abs(abs(M[1+i*int(len(T)/5),:])**2-abs(Man[1+i*int(len(T)/5),:])**2),"r:")#,label='t=0.001')
#
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"b:",label='$dt=0.01\cdot dx^2$')
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"g:",label='$dt=0.1dx^2$')
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"y:",label='$dt=2\cdot dx^2$')
#ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"r:",label='$dt=8\cdot dx^2$')
#
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"b:",label='$N=1000$')
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"g:",label='$N=500$')
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"y:",label='$N=300$')
##ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"r:",label='$N=100$')
#
#ax2.legend(loc='lower right')
#ax2.set_title( 'Odstopanja od točne rešitve pri razvoju začetnega kvantnega stanja skozi čas (n=2)')
#plt.xlabel('x')
#plt.ylabel('$||\psi_{an}|^2-|\psi_{nu}|^2|$')
#plt.xlim([0,1.3])
#plt.yscale('log')
#plt.show()



#FIGerrMU= plt.figure()
#MUnapake=plt.subplot(1, 1, 1 )
#
#for j in range(0,len(MU)):
#
#    U=pribl(MU[j],N)
#    GB, Niter, errGB =NEWTON(U,N,eps,delta)
#
#    if j==0:
#        MUnapake.plot(X, abs(GB-An), 'c:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('prva')
#    elif j==1:
#        MUnapake.plot(X, abs(GB-An), 'm:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('druga')
#    elif j==2:
#        MUnapake.plot(X, abs(GB-An), 'b:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('tretja')
#    elif j==3:
#        MUnapake.plot(X, abs(GB-An2), 'g:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('cetrta')
#    elif j==4:
#        MUnapake.plot(X, abs(GB-An2), 'r:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('peta')
#    elif j==5:
#        MUnapake.plot(X, abs(GB-An2), 'y:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('sesta')
#
#MUnapake.set_xlabel( '$x$' )
#MUnapake.set_ylabel( '$|GB_{num}-GB_{an}|$' )
#MUnapake.set_title( 'Natančnost diferenčne metode z $\delta=$'+str(delta)+' pri različnih $\mu$ in pogoju ob $err=$'+str(eps)+' (Newtonova)')
#MUnapake.legend(loc='upper right')
##MUnapake.legend(loc='best')
#MUnapake.set_yscale('log')
