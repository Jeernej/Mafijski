# -*- coding: utf-8 -*-

from math import *
from cmath import *
from scipy.special import *
from scipy.linalg import *
import numpy as np

#******************************************************
pi=3.1415926535897932384626433

def zacetno(x):
    return np.sin(pi*np.cos(x))

def analiticna(T, X):

    Man=np.zeros((len(T),len(X)))

    for i in range(len(T)):
        for l in range(len(X)):

            Man[i][l]=np.sin(pi*np.cos(X[l]+T[i])) #izračunam matriko analitičnih rešitev približek

    return Man

def dif1(dt,dx,N): # matrika propagacije začetnega približka v času
    
    A = np.zeros((N+1,N+1),np.complex_)
    
    for i in range(N):
        A[i][i+1] = dt/dx
    for i in range(N+1):
        A[i][i] = 1-(dt/dx)
        
    Az = A.conj()
    
    return A,Az


def razvoj(dx,dt,N,a, b,x0,dT): # eksplicitno v času

#    dx=(b-a)/N ## koraki na krajevnem intervalu
    X=[]
#    dt=6*dx**2 ## koraki na casovnem intervalu naredi kompromis
    T=[]
    T.append(0)

    psi0=[]
    for i in range(N+1) :
        psi0.append(zacetno(x0+i*dx)) #izračunam začetni približek
        X.append(x0+i*dx) #izračunam točke krajevnega intervala

    Mg=psi0  # v matriko časovnih korakov vpišem začetni približek ob času t=0
    A,Acon = dif1(dt,dx,N) # matrika propagacije začetnega približka v času


    cas=1
    k=0. #pogoj za vstop v zanko
    while dt*k<dT:

#        psi1 = np.linalg.solve(A,np.dot(Acon,psi0)) #rešim sistem za vrednost v novem časovnem koraku dt0
        psi1 = np.dot(A,psi0) #rešim sistem za vrednost v novem časovnem koraku dt0

        psi1[N] = psi1[0] #upoštevam periodični robni pogoj 
        
        if cas==1: Mg=np.concatenate(([Mg],[psi1]), axis=0) # v matriko časovnih korakov vpišem vrednost v novem časovnem koraku dt0
        else: Mg=np.concatenate((Mg,[psi1]), axis=0)

        psi0=psi1
        T.append(cas*dt) #računam točke časovnega intervala
        cas=cas+1
        k=k+1

    return T, X, Mg

#*********************** IZRAČUNI diferenčna ****************************


a=0
b=2*pi  # dolžina krajevnega intervala
x0=a
N=20 # delitev krajevnega intervala
dx=(b-a)/N # za N=100 0.06283185307179587

dT=4*pi# dolžina časovnega intervala

#dt=0.1  # za N=300 --> 0.14
#dt=0.05  # za N=300 --> 0.14
dt=0.0005  # za N=300 --> 0.14


T, X, M = razvoj(dx,dt,N,a, b,x0,dT)

Man = analiticna(T,X)

#********************** GRAFIRANJE ***************************

import matplotlib.pyplot as plt

from matplotlib import colors, ticker, cm

import matplotlib.ticker as mticker
def log_tick_formatter(val, pos=None):
    return "{:.0e}".format(10**val)
    
#**********************
## nalogi a.) in b.)
fig = plt.figure()
ax = fig.gca(projection='3d')
XX,TT = np.meshgrid(X,T)

#surf = ax.plot_wireframe(TT,XX,M, color = 'y', rstride=10, cstride=0, label='diferenčna')
surf = ax.plot_surface(TT,XX,M, rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0, label='diferenčna')
#surf = ax.plot_wireframe(TT,XX,Man, color = 'k', rstride=10, cstride=0, label='analitična')
#surf = ax.plot_surface(TT,XX,Man, rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0, label='analitična')

#surf = ax.plot_surface(TT,XX,abs(Man-M), rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0, label='odstopanje')

#surf = ax.plot_surface(TT,XX,np.log10(abs(Man-M)), rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0, label='odstopanje')
#surf = ax.plot_surface(MM,NN,np.log10(abs(0.757721-koeficient)), rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0)
#ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

#surf = ax.plot_surface(TT[0:150,:],XX[0:150,:],M[0:150,:], color = 'y', rstride=10, cstride=0, label='diferenčna')
#surf = ax.plot_wireframe(TT[0:150,:],XX[0:150,:],Man[0:150,:], color = 'k', rstride=10, cstride=0, label='analitična')
ax.set_title( 'Odstopanje med diferenčno in analitično rešitvijo hiperbolične valovne enačbe ($N_{x}=$'+str(N)+', $dt=$'+str(N)+')')
ax.set_title( 'Časovni razvoj za hiperbolično valovno enačbo (diferenčna rešitev $N_{x}=$'+str(N)+', $dt=$'+str(N)+')')
#ax.set_title( 'Časovni razvoj za hiperbolično valovno enačbo (analitična rešitev)')
ax.set_xlabel("t")
ax.set_ylabel("x")
#ax.set_zlabel("$u(x,t)$")
ax.set_zlabel("$|u_{dif}-u_{an}|$")
#ax.set_zlabel("$log_{10}|u_{dif}-u_{an}|$")
ax.set_ylim([0,2*pi])
#ax.legend(loc='best')
plt.tight_layout()
plt.show()




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

#*********************** IZRAČUNI ****************************

    
# izračun ak(0)

#def ak0(k,N,xkon = 2*pi):
#    u0,x0 = 0,0
#    h = xkon/N
#    for i in range(N):
#        u1,x1 = rk4_ak0(x0,u0,h,k)
#        u0,x0 = u1,x1
#    return u0
#
#def f_ak0(x,u,k):
#    return sin(pi*cos(x))*e**(-complex(0,1)*k*x)/(2*pi)
#
#def rk4_ak0(x,u,h,k):
#    k1 = h*f_ak0(x,u,k)
#    k2 = h*f_ak0(x+h/2,u+k1/2,k)
#    k3 = h*f_ak0(x+h/2,u+k2/2,k)
#    k4 = h*f_ak0(x+h,u+k3,k)
#    u = u + (k1+2*k2+2*k3+k4)/6
#    x += h
#    return u,x
#
## izračun ak(t)
#
#def akt(k,N,tkon):
#    h = tkon/N
#    a0,t0 = ak0(k,N),0
#    for i in range(N):
#        a1,t1 = rk4_akt(t0,a0,h,k)
#        a0,t0 = a1,t1
#    return a0
#
#def f_akt(t,a,k):
#    return complex(0,1)*k*a
#
#def rk4_akt(t,a,h,k):
#    k1 = h*f_akt(t,a,k)
#    k2 = h*f_akt(t+h/2,a+k1/2,k)
#    k3 = h*f_akt(t+h/2,a+k2/2,k)
#    k4 = h*f_akt(t+h,a+k3,k)
#    a = a + (k1+2*k2+2*k3+k4)/6
#    t += h
#    return a,t
#
## izračun hitrosti
#
#def fi(k,x):
#    return e**(complex(0,1)*k*x)
#
#def u(x,t,n,N):
#    vs = sum(fi(k,x)*akt(k,N,t) for k in range(-n/2,n/2+1,1))
#    return vs.real
#
## analticna funkcija
#
#def anal(x,t):
#    return sin(pi*cos(x+t))
#
## primerjava različnih metod:
## *analitičen izračun
## *diferenčna metoda
## *metoda Galerkina
#
#def dif_anal(tkon,xkon,Nt,Nx,zapI,zapJ):
#    zapis = open("dif_anal.txt","w")
#    k = tkon/Nt
#    h = xkon/Nx
#    u0 = [anal(j*h,0) for j in range(Nx+1)]
#    for i in range(Nt+1):
#        ref = u0[0]
#        for j in range(Nx):
#            if i%zapI == 0 and j%zapJ == 0:
#                zapis.write("{: > 010,.08e}  {: > 010,.08e}  {: > 024,.18f}  {: > 024,.18f}\n".format(i*k,j*h,anal(j*h,i*k),u0[j]))
#            if j==Nx:
#                u0[j] += (k/h)*(ref-u0[j])
#            else:
#                u0[j] += (k/h)*(u0[j+1]-u0[j])
#    zapis.close()
#
#def galerkin_anal(tkon,xkon,Nt,Nx,n,N):
#    zapis = open("galerkin_anal"+str(n)+".txt","w")
#    k = tkon/Nt
#    h = xkon/Nx
#
#    for i in range(Nt+1):
#        for j in range(Nx):
#            zapis.write("{: > 010,.08e}  {: > 010,.08e}  {: > 024,.18f}  {: > 024,.18f}\n".format(i*k,j*h,anal(j*h,i*k),u(j*h,i*k,n,N)))
#    zapis.close()
#    
#def aktanal(k,t):
#    return sin(k*pi/2)*jn(k,pi)*e**(complex(0,1)*k*t)
#
#"""
#zapis = open("aktR.txt","w")
#t=2.0
#for k in range(-30,30):
#    vr = aktanal(k,t).real
#    for i in range(20,501,5):
#        zapis.write("{: > 010,.08e}  {: > 010,.08e}  {: > 024,.18f}\n".format(k,i,log10(abs(vr-akt(k,i,t).real))))
#zapis.close()                                  
#"""
#print("bla")
#galerkin_anal(4.0,2*pi,40,40,8,50)
#print("bla")
#galerkin_anal(4.0,2*pi,40,40,16,100)
#print("bla")
#galerkin_anal(4.0,2*pi,40,40,32,100)
#print("bla")
#
#
