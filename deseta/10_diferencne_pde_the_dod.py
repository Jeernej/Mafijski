# -*- coding: utf-8 -*-
#from math import *
import scipy as sc
import numpy  as np
from numpy import linalg
from cmath import *

# *************************************************
pi=3.1415926535897932384626433

##square root of -1: Python adheres to this convention: a number followed by “j” is treated as an imaginary number!! i->1j

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
    
    
    
def difPade2(dt,dx,N,M,x0,k):
    
    A1 = np.zeros((N+1,N+1),np.complex_)
    A2 = np.zeros((N+1,N+1),np.complex_)
    A3 = np.zeros((N+1,N+1),np.complex_)
    A4 = np.zeros((N+1,N+1),np.complex_)

#    if M==1:
    k1 = -2.
    k2 = np.inf
    k3 = np.inf
    k4 = np.inf
    if M==2:
        k1 = -3.0 + 1j*1.73205
        k2 = k1.conjugate()
        k3 = np.inf
        k4 = np.inf
    elif M==3:
        k1 = -4.64437
        k2 = -3.67781 - 1j* 3.50876
        k3 = k2.conjugate()
        k4 = np.inf
    elif M==4:
        k1 = -4.20758 + 1j* 5.31484
        k2 = -5.79242 + 1j* 1.73447
        k3 = k2.conjugate()
        k4 = k1.conjugate()
        
#    b1,b2,b3,b4 = 1j*dt/(2.*dx**2),1j*dt/(2.*dx**2),1j*dt/(2.*dx**2),1j*dt/(2.*dx**2)  #probi brez 0.5
#    ee1,ee2,ee3,ee4 = -1./2*b2,-1./2*b2,-1./2*b3,-1./2*b4
    b=1j*dt/(2.*dx**2)
    c0,c1=-2.,1.
    ee1,ee2,ee3,ee4 = b*c1/k1,b*c1/k2,b*c1/k3,b*c1/k4

    for i in range(N):
        A1[i][i+1],A1[i+1][i] = ee1, ee1
        A2[i][i+1],A2[i+1][i] = ee2, ee2
        A3[i][i+1],A3[i+1][i] = ee3, ee3
        A4[i][i+1],A4[i+1][i] = ee4, ee4
    for i in range(N+1):
        A1[i][i] = 1 + b*c0/k1 - 1j*dt/k1*V(x0+i*dx,k)
        A2[i][i] = 1 + b*c0/k2 - 1j*dt/k2*V(x0+i*dx,k)
        A3[i][i] = 1 + b*c0/k3 - 1j*dt/k3*V(x0+i*dx,k)
        A4[i][i] = 1 + b*c0/k4 - 1j*dt/k4*V(x0+i*dx,k)        
       
    return A1,A2,A3,A4

    

def difPade4(dt,dx,N,M,x0,k):
    
    A1 = np.zeros((N+1,N+1),np.complex_)
    A2 = np.zeros((N+1,N+1),np.complex_)
    A3 = np.zeros((N+1,N+1),np.complex_)
    A4 = np.zeros((N+1,N+1),np.complex_)

#    if M==1:
    k1 = -2.
    k2 = np.inf
    k3 = np.inf
    k4 = np.inf
    if M==2:
        k1 = -3.0 + 1j*1.73205
        k2 = k1.conjugate()
        k3 = np.inf
        k4 = np.inf
    elif M==3:
        k1 = -4.64437
        k2 = -3.67781 - 1j* 3.50876
        k3 = k2.conjugate()
        k4 = np.inf
    elif M==4:
        k1 = -4.20758 + 1j* 5.31484
        k2 = -5.79242 + 1j* 1.73447
        k3 = k2.conjugate()
        k4 = k1.conjugate()

#    b1,b2,b3,b4 = 1j*dt/(k1*dx**2),1j*dt/(k2*dx**2),1j*dt/(k3*dx**2),1j*dt/(k4*dx**2)  #probi brez 0.5
#    ee1,ee2,ee3,ee4 = -4./6*b1,-4./6*b2,-4./6*b3,-4./6*b4
#    ff1,ff2,ff3,ff4 = 1./24*b1,1./24*b2,1./24*b3,1./24*b4
    b=1j*dt/(2.*dx**2)
    c0,c1,c2=-5./2,4./3,-1./12
    ee1,ee2,ee3,ee4 = b*c1/k1,b*c1/k2,b*c1/k3,b*c1/k4
    ff1,ff2,ff3,ff4 = b*c2/k1,b*c2/k2,b*c2/k3,b*c2/k4

    for i in range(N-1):
        A1[i][i+2],A1[i+2][i] = ff1, ff1
        A2[i][i+2],A2[i+2][i] = ff2, ff2
        A3[i][i+2],A3[i+2][i] = ff3, ff3
        A4[i][i+2],A4[i+2][i] = ff4, ff4
    for i in range(N):
        A1[i][i+1],A1[i+1][i] = ee1, ee1
        A2[i][i+1],A2[i+1][i] = ee2, ee2
        A3[i][i+1],A3[i+1][i] = ee3, ee3
        A4[i][i+1],A4[i+1][i] = ee4, ee4
    for i in range(N+1):
        A1[i][i] = 1 + b*c0/k1 - 1j*dt/k1*V(x0+i*dx,k)
        A2[i][i] = 1 + b*c0/k2 - 1j*dt/k2*V(x0+i*dx,k)
        A3[i][i] = 1 + b*c0/k3 - 1j*dt/k3*V(x0+i*dx,k)
        A4[i][i] = 1 + b*c0/k4 - 1j*dt/k4*V(x0+i*dx,k)        
       
    return A1,A2,A3,A4
    
    
def difPade6(dt,dx,N,M,x0,k):
    
    A1 = np.zeros((N+1,N+1),np.complex_)
    A2 = np.zeros((N+1,N+1),np.complex_)
    A3 = np.zeros((N+1,N+1),np.complex_)
    A4 = np.zeros((N+1,N+1),np.complex_)

#    if M==1:
    k1 = -2.
    k2 = np.inf
    k3 = np.inf
    k4 = np.inf
    if M==2:
        k1 = -3.0 + 1j*1.73205
        k2 = k1.conjugate()
        k3 = np.inf
        k4 = np.inf
    elif M==3:
        k1 = -4.64437
        k2 = -3.67781 - 1j* 3.50876
        k3 = k2.conjugate()
        k4 = np.inf
    elif M==4:
        k1 = -4.20758 + 1j* 5.31484
        k2 = -5.79242 + 1j* 1.73447
        k3 = k2.conjugate()
        k4 = k1.conjugate()

#    b1,b2,b3,b4 = 1j*dt/(k1*dx**2),1j*dt/(k2*dx**2),1j*dt/(k3*dx**2),1j*dt/(k4*dx**2)  #probi brez 0.5
#    ee1,ee2,ee3,ee4 = -3.0/4*b1,-3.0/4*b2,-3.0/4*b3,-3.0/4*b4
#    ff1,ff2,ff3,ff4 = 3.0/40*b1,3.0/40*b2,3.0/40*b3,3.0/40*b4
#    gg1,gg2,gg3,gg4 = -1.0/180*b1,-1.0/180*b2,-1.0/180*b3,-1.0/180*b4
    b=1j*dt/(2.*dx**2)
    c0,c1,c2,c3=-49./18,3./2,-3./20,1./90
    ee1,ee2,ee3,ee4 = b*c1/k1,b*c1/k2,b*c1/k3,b*c1/k4
    ff1,ff2,ff3,ff4 = b*c2/k1,b*c2/k2,b*c2/k3,b*c2/k4
    gg1,gg2,gg3,gg4 = b*c3/k1,b*c3/k2,b*c3/k3,b*c3/k4
    
    for i in range(N-2):
        A1[i][i+3],A1[i+3][i] = gg1, gg1
        A2[i][i+3],A2[i+3][i] = gg2, gg2
        A3[i][i+3],A3[i+3][i] = gg3, gg3
        A4[i][i+3],A4[i+3][i] = gg4, gg4
    for i in range(N-1):
        A1[i][i+2],A1[i+2][i] = ff1, ff1
        A2[i][i+2],A2[i+2][i] = ff2, ff2
        A3[i][i+2],A3[i+2][i] = ff3, ff3
        A4[i][i+2],A4[i+2][i] = ff4, ff4
    for i in range(N):
        A1[i][i+1],A1[i+1][i] = ee1, ee1
        A2[i][i+1],A2[i+1][i] = ee2, ee2
        A3[i][i+1],A3[i+1][i] = ee3, ee3
        A4[i][i+1],A4[i+1][i] = ee4, ee4
    for i in range(N+1):
        A1[i][i] = 1 + b*c0/k1 - 1j*dt/k1*V(x0+i*dx,k)
        A2[i][i] = 1 + b*c0/k2 - 1j*dt/k2*V(x0+i*dx,k)
        A3[i][i] = 1 + b*c0/k3 - 1j*dt/k3*V(x0+i*dx,k)
        A4[i][i] = 1 + b*c0/k4 - 1j*dt/k4*V(x0+i*dx,k)
       
    return A1,A2,A3,A4

#def diferenca6Pade(dt,dx,x0,N):
#    A1 = np.zeros((N+1,N+1),np.complex_)
#    A2 = np.zeros((N+1,N+1),np.complex_)
#    k1 = -3.0 + complex(0,1)*1.73205
#    k2 = k1.conjugate()
#    
#    b1,b2 = complex(0,1)*dt/(k1*dx**2),complex(0,1)*dt/(k2*dx**2)
#    ee1,ee2 = -3.0/4*b1,-3.0/4*b2
#    ff1,ff2 = 3.0/40*b1,3.0/40*b2
#    gg1,gg2 = -1.0/180*b1,-1.0/180*b2
#    for i in range(N-2):
#        A1[i][i+3],A1[i+3][i] = gg1, gg1
#        A2[i][i+3],A2[i+3][i] = gg2, gg2
#    for i in range(N-1):
#        A1[i][i+2],A1[i+2][i] = ff1, ff1
#        A2[i][i+2],A2[i+2][i] = ff2, ff2
#    for i in range(N):
#        A1[i][i+1],A1[i+1][i] = ee1, ee1
#        A2[i][i+1],A2[i+1][i] = ee2, ee2
#    for i in range(N+1):
#        A1[i][i] = 1 + 49.0/36*b1 + complex(0,1)*dt/k1*Vpot(x0+i*dx)
#        A2[i][i] = 1 + 49.0/36*b2 + complex(0,1)*dt/k2*Vpot(x0+i*dx)
#    
#    return A1,A2
def difPade8(dt,dx,N,M,x0,k):
    
    A1 = np.zeros((N+1,N+1),np.complex_)
    A2 = np.zeros((N+1,N+1),np.complex_)
    A3 = np.zeros((N+1,N+1),np.complex_)
    A4 = np.zeros((N+1,N+1),np.complex_)

#    if M==1:
    k1 = -2.
    k2 = np.inf
    k3 = np.inf
    k4 = np.inf
    if M==2:
        k1 = -3.0 + 1j*1.73205
        k2 = k1.conjugate()
        k3 = np.inf
        k4 = np.inf
    elif M==3:
        k1 = -4.64437
        k2 = -3.67781 - 1j* 3.50876
        k3 = k2.conjugate()
        k4 = np.inf
    elif M==4:
        k1 = -4.20758 + 1j* 5.31484
        k2 = -5.79242 + 1j* 1.73447
        k3 = k2.conjugate()
        k4 = k1.conjugate()        
        
#    b1,b2,b3,b4 = 1j*dt/(k1*dx**2),1j*dt/(k2*dx**2),1j*dt/(k3*dx**2),1j*dt/(k4*dx**2)  #probi brez 0.5
#    ee1,ee2,ee3,ee4 = -8./10*b1,-8./10*b2,-8./10*b3,-8./10*b4
#    ff1,ff2,ff3,ff4 = 1./10*b1,1./10*b2,1./10*b3,1./10*b4
#    gg1,gg2,gg3,gg4 = -8./630*b1,-8./630*b2,-8./630*b3,-8./630*b4
#    hh1,hh2,hh3,hh4 =  1./1120*b1, 1./1120*b2, 1./1120*b3, 1./1120*b4
    b=1j*dt/(2.*dx**2)
    c0,c1,c2,c3,c4=-205./72,8./5,-1./5,8./315,-1./560
    ee1,ee2,ee3,ee4 = b*c1/k1,b*c1/k2,b*c1/k3,b*c1/k4
    ff1,ff2,ff3,ff4 = b*c2/k1,b*c2/k2,b*c2/k3,b*c2/k4
    gg1,gg2,gg3,gg4 = b*c3/k1,b*c3/k2,b*c3/k3,b*c3/k4
    hh1,hh2,hh3,hh4 = b*c4/k1,b*c4/k2,b*c4/k3,b*c4/k4
    
    for i in range(N-3):
        A1[i][i+4],A1[i+4][i] = hh1, hh1    
        A2[i][i+4],A2[i+4][i] = hh2, hh2    
        A3[i][i+4],A3[i+4][i] = hh3, hh3    
        A4[i][i+4],A4[i+4][i] = hh4, hh4    
    for i in range(N-2):
        A1[i][i+3],A1[i+3][i] = gg1, gg1
        A2[i][i+3],A2[i+3][i] = gg2, gg2
        A3[i][i+3],A3[i+3][i] = gg3, gg3
        A4[i][i+3],A4[i+3][i] = gg4, gg4
    for i in range(N-1):
        A1[i][i+2],A1[i+2][i] = ff1, ff1
        A2[i][i+2],A2[i+2][i] = ff2, ff2
        A3[i][i+2],A3[i+2][i] = ff3, ff3
        A4[i][i+2],A4[i+2][i] = ff4, ff4
    for i in range(N):
        A1[i][i+1],A1[i+1][i] = ee1, ee1
        A2[i][i+1],A2[i+1][i] = ee2, ee2
        A3[i][i+1],A3[i+1][i] = ee3, ee3
        A4[i][i+1],A4[i+1][i] = ee4, ee4
    for i in range(N+1):
        A1[i][i] = 1 + b*c0/k1 - 1j*dt/k1*V(x0+i*dx,k)
        A2[i][i] = 1 + b*c0/k2 - 1j*dt/k2*V(x0+i*dx,k) 
        A3[i][i] = 1 + b*c0/k3 - 1j*dt/k3*V(x0+i*dx,k)
        A4[i][i] = 1 + b*c0/k4 - 1j*dt/k4*V(x0+i*dx,k)
       
    return A1,A2,A3,A4

def razvojPade(dx,dt,N,a, b, k0,x0,dT,alfa,lam, diff,M):
    
#    dx=(b-a)/N ## koraki na krajevnem intervalu dx€[0,0.75]
    X=[]
#    dt=6*dx**2 ## koraki na casovnem intervalu
    T=[]
    T.append(0)

    psi0=[]
    for i in range(N+1) :
        psi0.append(zacetno(x0+i*dx, alfa, lam )) #izračunam začetni približek
        X.append(x0+i*dx) #izračunam točke krajevnega intervala

    Mg=psi0  # v matriko časovnih korakov vpišem začetni približek ob času t=0
    
#    if diff==2:
    A1,A2,A3,A4 = difPade2(dt,dx,N,M,x0,k0)
    if diff==4:
        A1,A2,A3,A4 = difPade4(dt,dx,N,M,x0,k0)
    elif diff==6:
        A1,A2,A3,A4 = difPade6(dt,dx,N,M,x0,k0)
    elif diff==8:
        A1,A2,A3,A4 = difPade8(dt,dx,N,M,x0,k0)

    cas=1
    k=0. #pogoj za vstop v zanko
    while dt*k/M<10*dT:
        
        psi1 = np.linalg.solve(A1,np.dot(A1.conj(),psi0))
        psi0 = psi1
        
        if M==2:
            psi2 = np.linalg.solve(A2,np.dot(A2.conj(),psi1))
            psi0 = psi2
        elif M==3:
            psi2 = np.linalg.solve(A2,np.dot(A2.conj(),psi1))
            psi3 = np.linalg.solve(A3,np.dot(A3.conj(),psi2))
            psi0 = psi3
        elif M==4:
            psi2 = np.linalg.solve(A2,np.dot(A2.conj(),psi1))
            psi3 = np.linalg.solve(A3,np.dot(A3.conj(),psi2))
            psi4 = np.linalg.solve(A4,np.dot(A4.conj(),psi3))
            psi0 = psi4
            
        if cas==1: Mg=np.concatenate(([Mg],[psi0]), axis=0) # v matriko časovnih korakov vpišem vrednost v novem časovnem koraku dt0
        else: Mg=np.concatenate((Mg,[psi0]), axis=0) 
        
        psi0=psi1
        T.append(cas*dt/M) #računam točke časovnega intervala
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
    

def difPadeGauss2(dt,dx,N,M):
    
    A1 = np.zeros((N+1,N+1),np.complex_)
    A2 = np.zeros((N+1,N+1),np.complex_)
    A3 = np.zeros((N+1,N+1),np.complex_)
    A4 = np.zeros((N+1,N+1),np.complex_)

#    if M==1:
    k1 = -2.
    k2 = np.inf
    k3 = np.inf
    k4 = np.inf
    if M==2:
        k1 = -3.0 + 1j*1.73205
        k2 = k1.conjugate()
        k3 = np.inf
        k4 = np.inf
    elif M==3:
        k1 = -4.64437
        k2 = -3.67781 - 1j* 3.50876
        k3 = k2.conjugate()
        k4 = np.inf
    elif M==4:
        k1 = -4.20758 + 1j* 5.31484
        k2 = -5.79242 + 1j* 1.73447
        k3 = k2.conjugate()
        k4 = k1.conjugate()
        
    b1,b2,b3,b4 = -1j*dt/(k1*dx**2),-1j*dt/(k2*dx**2),-1j*dt/(k3*dx**2),-1j*dt/(k4*dx**2)  # morjo bit negativni (pojma nimam zakaj)
    ee1,ee2,ee3,ee4 = -1./2*b1,-1./2*b2,-1./2*b3,-1./2*b4

    for i in range(N):
        A1[i][i+1],A1[i+1][i] = ee1, ee1
        A2[i][i+1],A2[i+1][i] = ee2, ee2
        A3[i][i+1],A3[i+1][i] = ee3, ee3
        A4[i][i+1],A4[i+1][i] = ee4, ee4
    for i in range(N+1):
        A1[i][i] = 1 + b1
        A2[i][i] = 1 + b2
        A3[i][i] = 1 + b3
        A4[i][i] = 1 + b4        
       
    return A1,A2,A3,A4

    

def difPadeGauss4(dt,dx,N,M):
    
    A1 = np.zeros((N+1,N+1),np.complex_)
    A2 = np.zeros((N+1,N+1),np.complex_)
    A3 = np.zeros((N+1,N+1),np.complex_)
    A4 = np.zeros((N+1,N+1),np.complex_)

#    if M==1:
    k1 = -2.
    k2 = np.inf
    k3 = np.inf
    k4 = np.inf
    if M==2:
        k1 = -3.0 + 1j*1.73205
        k2 = k1.conjugate()
        k3 = np.inf
        k4 = np.inf
    elif M==3:
        k1 = -4.64437
        k2 = -3.67781 - 1j* 3.50876
        k3 = k2.conjugate()
        k4 = np.inf
    elif M==4:
        k1 = -4.20758 + 1j* 5.31484
        k2 = -5.79242 + 1j* 1.73447
        k3 = k2.conjugate()
        k4 = k1.conjugate()

    b1,b2,b3,b4 = -1j*dt/(k1*dx**2),-1j*dt/(k2*dx**2),-1j*dt/(k3*dx**2),-1j*dt/(k4*dx**2)  # morjo bit negativni (pojma nimam zakaj)
    ee1,ee2,ee3,ee4 = -4./6*b1,-4./6*b2,-4./6*b3,-4./6*b4
    ff1,ff2,ff3,ff4 = 1./24*b1,1./24*b2,1./24*b3,1./24*b4

    for i in range(N-1):
        A1[i][i+2],A1[i+2][i] = ff1, ff1
        A2[i][i+2],A2[i+2][i] = ff2, ff2
        A3[i][i+2],A3[i+2][i] = ff3, ff3
        A4[i][i+2],A4[i+2][i] = ff4, ff4
    for i in range(N):
        A1[i][i+1],A1[i+1][i] = ee1, ee1
        A2[i][i+1],A2[i+1][i] = ee2, ee2
        A3[i][i+1],A3[i+1][i] = ee3, ee3
        A4[i][i+1],A4[i+1][i] = ee4, ee4
    for i in range(N+1):
        A1[i][i] = 1 + 5./4*b1
        A2[i][i] = 1 + 5./4*b2
        A3[i][i] = 1 + 5./4*b3
        A4[i][i] = 1 + 5./4*b4        
       
    return A1,A2,A3,A4
    
    
def difPadeGauss6(dt,dx,N,M):
    
    A1 = np.zeros((N+1,N+1),np.complex_)
    A2 = np.zeros((N+1,N+1),np.complex_)
    A3 = np.zeros((N+1,N+1),np.complex_)
    A4 = np.zeros((N+1,N+1),np.complex_)

#    if M==1:
    k1 = -2.
    k2 = np.inf
    k3 = np.inf
    k4 = np.inf
    if M==2:
        k1 = -3.0 + 1j*1.73205
        k2 = k1.conjugate()
        k3 = np.inf
        k4 = np.inf
    elif M==3:
        k1 = -4.64437
        k2 = -3.67781 - 1j* 3.50876
        k3 = k2.conjugate()
        k4 = np.inf
    elif M==4:
        k1 = -4.20758 + 1j* 5.31484
        k2 = -5.79242 + 1j* 1.73447
        k3 = k2.conjugate()
        k4 = k1.conjugate()

    b1,b2,b3,b4 = -1j*dt/(k1*dx**2),-1j*dt/(k2*dx**2),-1j*dt/(k3*dx**2),-1j*dt/(k4*dx**2)  # morjo bit negativni (pojma nimam zakaj)
    ee1,ee2,ee3,ee4 = -3.0/4*b1,-3.0/4*b2,-3.0/4*b3,-3.0/4*b4
    ff1,ff2,ff3,ff4 = 3.0/40*b1,3.0/40*b2,3.0/40*b3,3.0/40*b4
    gg1,gg2,gg3,gg4 = -1.0/180*b1,-1.0/180*b2,-1.0/180*b3,-1.0/180*b4
    
    for i in range(N-2):
        A1[i][i+3],A1[i+3][i] = gg1, gg1
        A2[i][i+3],A2[i+3][i] = gg2, gg2
        A3[i][i+3],A3[i+3][i] = gg3, gg3
        A4[i][i+3],A4[i+3][i] = gg4, gg4
    for i in range(N-1):
        A1[i][i+2],A1[i+2][i] = ff1, ff1
        A2[i][i+2],A2[i+2][i] = ff2, ff2
        A3[i][i+2],A3[i+2][i] = ff3, ff3
        A4[i][i+2],A4[i+2][i] = ff4, ff4
    for i in range(N):
        A1[i][i+1],A1[i+1][i] = ee1, ee1
        A2[i][i+1],A2[i+1][i] = ee2, ee2
        A3[i][i+1],A3[i+1][i] = ee3, ee3
        A4[i][i+1],A4[i+1][i] = ee4, ee4
    for i in range(N+1):
        A1[i][i] = 1 + 49.0/36*b1
        A2[i][i] = 1 + 49.0/36*b2
        A3[i][i] = 1 + 49.0/36*b3
        A4[i][i] = 1 + 49.0/36*b4
       
    return A1,A2,A3,A4
    

def difPadeGauss8(dt,dx,N,M):
    
    A1 = np.zeros((N+1,N+1),np.complex_)
    A2 = np.zeros((N+1,N+1),np.complex_)
    A3 = np.zeros((N+1,N+1),np.complex_)
    A4 = np.zeros((N+1,N+1),np.complex_)

#    if M==1:
    k1 = -2.
    k2 = np.inf
    k3 = np.inf
    k4 = np.inf
    if M==2:
        k1 = -3.0 + 1j*1.73205
        k2 = k1.conjugate()
        k3 = np.inf
        k4 = np.inf
    elif M==3:
        k1 = -4.64437
        k2 = -3.67781 - 1j* 3.50876
        k3 = k2.conjugate()
        k4 = np.inf
    elif M==4:
        k1 = -4.20758 + 1j* 5.31484
        k2 = -5.79242 + 1j* 1.73447
        k3 = k2.conjugate()
        k4 = k1.conjugate()        
        
    b1,b2,b3,b4 = -1j*dt/(k1*dx**2),-1j*dt/(k2*dx**2),-1j*dt/(k3*dx**2),-1j*dt/(k4*dx**2)  # morjo bit negativni (pojma nimam zakaj)
    ee1,ee2,ee3,ee4 = -8./10*b1,-8./10*b2,-8./10*b3,-8./10*b4
    ff1,ff2,ff3,ff4 = 1./10*b1,1./10*b2,1./10*b3,1./10*b4
    gg1,gg2,gg3,gg4 = -8./630*b1,-8./630*b2,-8./630*b3,-8./630*b4
    hh1,hh2,hh3,hh4 =  1./1120*b1, 1./1120*b2, 1./1120*b3, 1./1120*b4


    for i in range(N-3):
        A1[i][i+4],A1[i+4][i] = hh1, hh1    
        A2[i][i+4],A2[i+4][i] = hh2, hh2    
        A3[i][i+4],A3[i+4][i] = hh3, hh3    
        A4[i][i+4],A4[i+4][i] = hh4, hh4    
    for i in range(N-2):
        A1[i][i+3],A1[i+3][i] = gg1, gg1
        A2[i][i+3],A2[i+3][i] = gg2, gg2
        A3[i][i+3],A3[i+3][i] = gg3, gg3
        A4[i][i+3],A4[i+3][i] = gg4, gg4
    for i in range(N-1):
        A1[i][i+2],A1[i+2][i] = ff1, ff1
        A2[i][i+2],A2[i+2][i] = ff2, ff2
        A3[i][i+2],A3[i+2][i] = ff3, ff3
        A4[i][i+2],A4[i+2][i] = ff4, ff4
    for i in range(N):
        A1[i][i+1],A1[i+1][i] = ee1, ee1
        A2[i][i+1],A2[i+1][i] = ee2, ee2
        A3[i][i+1],A3[i+1][i] = ee3, ee3
        A4[i][i+1],A4[i+1][i] = ee4, ee4
    for i in range(N+1):
        A1[i][i] = 1 + 205./144*b1
        A2[i][i] = 1 + 205./144*b2
        A3[i][i] = 1 + 205./144*b3
        A4[i][i] = 1 + 205./144*b4
       
    return A1,A2,A3,A4
    

    
def razvojPadeGauss(dx,dt,N,a0, b0, k0, sig0, lamb, diff,M):
    
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
    
    
#    if diff==2:
    A1,A2,A3,A4 = difPadeGauss2(dt,dx,N,M)
    if diff==4:
        A1,A2,A3,A4 = difPadeGauss4(dt,dx,N,M)
    elif diff==6:
        A1,A2,A3,A4 = difPadeGauss6(dt,dx,N,M)
    elif diff==8:
        A1,A2,A3,A4 = difPadeGauss8(dt,dx,N,M)

    cas=1
    tez=0. #pogoj za vstop v zanko
    while tez < 0.75:
        
        psi1 = np.linalg.solve(A1,np.dot(A1.conj(),psi0))
        psi0 = psi1
        
        if M==2:
            psi2 = np.linalg.solve(A2,np.dot(A2.conj(),psi1))
            psi0 = psi2
        elif M==3:
            psi2 = np.linalg.solve(A2,np.dot(A2.conj(),psi1))
            psi3 = np.linalg.solve(A3,np.dot(A3.conj(),psi2))
            psi0 = psi3
        elif M==4:
            psi2 = np.linalg.solve(A2,np.dot(A2.conj(),psi1))
            psi3 = np.linalg.solve(A3,np.dot(A3.conj(),psi2))
            psi4 = np.linalg.solve(A4,np.dot(A4.conj(),psi3))
            psi0 = psi4
            
        if cas==1: Mg=np.concatenate(([Mg],[psi0]), axis=0) # v matriko časovnih korakov vpišem vrednost v novem časovnem koraku dt0
        else: Mg=np.concatenate((Mg,[psi0]), axis=0) 
        
        tez = sum([dx*(a0+(i + 0.5)*dx)*(abs(psi0[i])**2+abs(psi0[i + 1])**2)/2. for i in range(N-1)])

        T.append(cas*dt) #računam točke časovnega intervala
        cas=cas+1

    return T, X, Mg



#*********************** IZRAČUNI ****************************

#N=50

#N=100
N=300
#N=500

#N=1000

diff=8
# ****************** paket v potencialu  *******************************

#k = 0.04
#lam = 10.
#w = np.sqrt(k)
#alfa = k**0.25
#dT=2.*pi/w
#
#a=-40.
#b=40.
#x0=-40.
#dx=(b-a)/N
#
##dt=0.14
##dt=0.1*dx**2 # za N=300 --> 0.007
##dt=dx**2  # za N=300 --> 0.07
#dt=2*dx**2  # za N=300 --> 0.14
##dt=4*dx**2 # za N=300 --> 0.285
##dt=8*dx**2 # za N=300 --> 0.57
#
## razvojPade(N,a, b, k0,x0,dT,alfa,lam, diff,M)
#T, X, M = razvojPade(dx,dt,N,a, b, k, x0,dT,alfa,lam,diff, 1)
#T4, X4, M4 = razvojPade(dx,dt,N,a, b, k, x0,dT,alfa,lam,diff,2)
#T6, X6, M6 = razvojPade(dx,dt,N,a, b, k, x0,dT,alfa,lam,diff,3)
#T8, X8, M8 = razvojPade(dx,dt,N,a, b, k, x0,dT,alfa,lam,diff,4)
#
#Man = analiticna(T,X, alfa, lam, w)

# ****************** Gausovski paket  brez potenciala  *******************************

k0 = 50*pi
sig0 = 1./20
lamb = 0.25

a0=-0.5
b0=1.5
dx=(b0-a0)/N

#dt=8.88888888888889*10**(-5)
#dt=0.01*dx**2 # za N=300 -->
#dt=0.1*dx**2 # za N=300 -->
dt=2*dx**2  # za N=300 --> 8.88888888888889e-05
#dt=8*dx**2 # za N=300 --> 110.44661672776617

T, X, M= razvojPadeGauss(dx,dt,N,a0,b0,k0,sig0,lamb,diff,1)
T4, X4, M4 = razvojPadeGauss(dx,dt,N,a0,b0,k0,sig0,lamb,diff,2)
T6, X6, M6 = razvojPadeGauss(dx,dt,N,a0,b0,k0,sig0,lamb,diff,3)
T8, X8, M8 = razvojPadeGauss(dx,dt,N,a0,b0,k0,sig0,lamb,diff,4)

Man = analiticnaGauss(T,X, sig0, k0, lamb)

#********************** GRAFIRANJE ***************************

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


## nalogi a.) in b.)
fig = plt.figure()
ax = fig.gca(projection='3d')
XX,TT = np.meshgrid(X,T)
XX4,TT4 = np.meshgrid(X4,T4)
XX6,TT6 = np.meshgrid(X6,T6)
XX8,TT8 = np.meshgrid(X8,T8)

surf = ax.plot_wireframe(TT,XX,abs(M)**2, color = 'y', rstride=10, cstride=0, label='M=1')
surf = ax.plot_wireframe(TT4,XX4,abs(M4)**2, color = 'blue', rstride=10, cstride=0, label='M=2')
surf = ax.plot_wireframe(TT6,XX6,abs(M6)**2, color = 'green', rstride=10, cstride=0,label='M=3')
surf = ax.plot_wireframe(TT8,XX8,abs(M8)**2, color = 'red', rstride=10, cstride=0, label='M=4')
surf = ax.plot_wireframe(TT,XX,abs(Man)**2, color = 'k', rstride=10, cstride=0, label='analitična')

#surf = ax.plot_wireframe(TT[0:150,:],XX[0:150,:],abs(M[0:150,:])**2, color = 'y', rstride=10, cstride=0, label='M=1')
#surf = ax.plot_wireframe(TT4[0:150*2,:],XX4[0:150*2,:],abs(M4[0:150*2,:])**2, color = 'blue', rstride=10, cstride=0, label='M=2')
#surf = ax.plot_wireframe(TT6[0:150*3,:],XX6[0:150*3,:],abs(M6[0:150*3,:])**2, color = 'green', rstride=10, cstride=0,label='M=3')
#surf = ax.plot_wireframe(TT8[0:150*5,:],XX8[0:150*5,:],abs(M8[0:150*5,:])**2, color = 'red', rstride=10, cstride=0, label='M=4')
#surf = ax.plot_wireframe(TT[0:150,:],XX[0:150,:],abs(Man[0:150,:])**2, color = 'k', rstride=10, cstride=0, label='analitična')


#ax.plot_surface(TT, XX, M, rstride=8, cstride=8, alpha=0.3)
#cset = ax.contourf(TT, XX, M, zdir='m', offset=-100, cmap=cm.coolwarm)
#cset = ax.contourf(TT, XX, M, zdir='t', offset=-40, cmap=cm.coolwarm)
#cset = ax.contourf(TT, XX, M, zdir='y', offset=40, cmap=cm.coolwarm)
ax.set_title( 'Časovni razvoj začetnega kvantnega stanja (n='+str(diff)+')')
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_zlabel("$|\psi(x,t)|^2$")
#ax.set_ylim([-20,20])
ax.legend(loc='best')
plt.tight_layout()
plt.show()


## naloga a.)
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
##for i in range(int(7)):
##    ax1.plot(X,abs(M[i*int(dT/dt/2/7),:])**2,"y--")#,label='t=0.001')
##    ax1.plot(X4,abs(M4[i*int(dT/dt/2/7),:])**2,"y--")#,label='t=0.001')
##    ax1.plot(X6,abs(M6[i*int(dT/dt/2/7),:])**2,"y--")#,label='t=0.001')
##    ax1.plot(X8,abs(M8[i*int(dT/dt/2/7),:])**2,"y--")#,label='t=0.001')
##    ax1.plot(X,abs(Man[i*int(dT/dt/2/7),:])**2,"k--")#,label='t=0.001')
##    ax1.plot(X,abs(M[2209-(i+1)*int(dT/dt/2/7),:])**2,"y:")#,label='t=0.001')
##    ax1.plot(X4,abs(M4[2209-(i+1)*int(dT/dt/2/7),:])**2,"y:")#,label='t=0.001')
##    ax1.plot(X6,abs(M6[2209-(i+1)*int(dT/dt/2/7),:])**2,"y:")#,label='t=0.001')
##    ax1.plot(X8,abs(M8[2209-(i+1)*int(dT/dt/2/7),:])**2,"y:")#,label='t=0.001')
##    ax1.plot(X,abs(Man[2209-(i+1)*int(dT/dt/2/7),:])**2,"k:")#,label='t=0.001')
#
#ax1.plot(X,abs(M[8*int(dT/dt/2/7),:])**2,"y--",label='M=1; (0<t<T/2)')
#ax1.plot(X4,abs(M4[8*int(dT/dt/2/7)*2,:])**2,"b--",label='M=2; (0<t<T/2)')
#ax1.plot(X6,abs(M6[8*int(dT/dt/2/7)*3,:])**2,"g--",label='M=3; (0<t<T/2)')
#ax1.plot(X8,abs(M8[8*int(dT/dt/2/7)*4,:])**2,"r--",label='M=4; (0<t<T/2)')
#ax1.plot(X,abs(Man[8*int(dT/dt/2/7),:])**2,"k--",label='analiticna; (0<t<T/2)')
#
#ax1.plot(X,abs(M[2209,:])**2,"y:",label='M=1; (19T/2<t<10T)')
#ax1.plot(X,abs(M4[2209*2,:])**2,"b:",label='M=2; (19T/2<t<10T)')
#ax1.plot(X,abs(M6[2209*3,:])**2,"g:",label='M=3; (19T/2<t<10T)')
#ax1.plot(X,abs(M8[2209*4,:])**2,"r:",label='M=4; (19T/2<t<10T)')
#ax1.plot(X,abs(Man[2209,:])**2,"k:",label='analiticna; (19T/2<t<10T)')
#
##ax1.plot(X,M[0,:],"k",label='t=0.001')
##ax1.plot(X,M[100,:],"k--",label='t=0.1')
##ax1.plot(X,M[100000,:],"k-.",label='t=100')
##ax1.plot(X,M[200000,:],"k:",label='t=200')
##ax1.plot(X,M[400000,:],"k:",label='t=200')
#ax1.legend(loc='best')
#ax1.set_title( 'Časovni razvoj začetnega kvantnega stanja (n='+str(diff)+')')
#plt.xlabel('x')
#plt.ylabel('$|\psi(x,t)|^2$')
#plt.xlim([-20,20])
#plt.show()
#


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

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
for i in range(int(4)):
    ax2.plot(X,abs(abs(M[1+i*int(len(T)/5),:])**2-abs(Man[1+i*int(len(T)/5),:])**2),"b:")#,label='t=0.001')
    ax2.plot(X,abs(abs(M4[1+i*int(len(T4)/5),:])**2-abs(Man[1+i*int(len(T)/5),:])**2),"g:")#,label='t=0.001')
    ax2.plot(X,abs(abs(M6[1+i*int(len(T6)/5),:])**2-abs(Man[1+i*int(len(T)/5),:])**2),"y:")#,label='t=0.001')
    ax2.plot(X,abs(abs(M7[1+i*int(len(T8)/5),:])**2-abs(Man[1+i*int(len(T)/5),:])**2),"r:")#,label='t=0.001')

#ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T)-1,:])**2),"b:",label='$dt=0.01\cdot dx^2$')
#ax2.plot(X,abs(abs(M[len(T6)-1,:])**2-abs(Man[len(T)-1,:])**2),"g:",label='$dt=0.1dx^2$')
#ax2.plot(X,abs(abs(M[len(T4)-1,:])**2-abs(Man[len(T)-1,:])**2),"y:",label='$dt=2\cdot dx^2$')
#ax2.plot(X,abs(abs(M[len(T8)-1,:])**2-abs(Man[len(T)-1,:])**2),"r:",label='$dt=8\cdot dx^2$')

ax2.plot(X,abs(abs(M[len(T)-1,:])**2-abs(Man[len(T8)-1,:])**2),"b:",label='$M=1$')
ax2.plot(X,abs(abs(M4[len(T4)-1,:])**2-abs(Man[len(T8)-1,:])**2),"g:",label='$M=2$')
ax2.plot(X,abs(abs(M6[len(T6)-1,:])**2-abs(Man[len(T8)-1,:])**2),"y:",label='$M=3')
ax2.plot(X,abs(abs(M8[len(T8)-1,:])**2-abs(Man[len(T8)-1,:])**2),"r:",label='$M=4$')

ax2.legend(loc='lower right')
ax2.set_title( 'Odstopanja od točne rešitve pri razvoju začetnega kvantnega stanja skozi čas (n='+str(diff)+')')
plt.xlabel('x')
plt.ylabel('$||\psi_{an}|^2-|\psi_{nu}|^2|$')
plt.xlim([0,1.3])
plt.yscale('log')
plt.show()

