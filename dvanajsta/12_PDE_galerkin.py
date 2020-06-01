# -*- coding: utf-8 -*-


import numpy  as np
import scipy as sc
from numpy import linalg
from scipy import special

#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#from pylab import *

#******************************************************
pi=3.1415926535897932384626433

def Phimn(x,phi,m,n):
    
    Phi=[]
   
    for i in range(m+1):
        for j in range(1,n+1,1):    
            Phi.append( x**(2*i + 1.)*(1. - x)**j *sc.sin((2.*i+1)*phi) )

    return Phi



def bj(m,n):     # vektor bj

    b=[]
    
    for i in range(m+1):
        for j in range(1,n+1,1):
            
            b.append(-2.*sc.special.beta(2*i+3,j+1) / (2*i+1))   # vektor velikosti m'xn' , kjer najprej za vsak m' štejemo po vseh n'
#            b.append(2.*(i+1))   # vektor velikosti m'xn' , kjer najprej za vsak m' štejemo po vseh n'

    return b
    
    

def Aij(m,n):     # bločna matrika Aij z mxm' bloki (vsak dimenzije nxn') različnimi od 0 le na diagonali, kjer je m=m'

    A=np.zeros(((m+1)*n,(m+1)*n)) # A[vrstica][stolpec]
    
    for i in range(m+1):
        for j in range(m+1):           
            
            if i==j: #sedaj smo znotraj enega bloka (pogoj za kroneckerjev delta m=m')

                for k in range(1,n+1):
                    for l in range(1,n+1):   #sedaj smo v bloku 
                        
                        A[n*i+l-1][n*j+k-1]=-pi*k*l*(3+4.*i)*sc.special.beta(k+l-1.,3+4.*i) /2./(2.+4.*i+k+l)
#                        A[n*j+l-1][n*i+k-1]=(i+1)*2.
                        
    return A
    
    

#def C(m, n):
#    
#    A=Aij(m,n)
#    b=bj(m,n)
#    a=np.linalg.solve(A,b)
#
#            
#    C=(-32./pi)*sum([b[i]*a[i] for i in range(len(a))])   
#              
#    return M,N,C

#
def C(m, n):
    
    C=np.zeros((m+1,n+1))    
 
    for i in range(m+1):
        print(i)      
        for j in range(1,n+1):
            
            A=Aij(i,j)
            b=bj(i,j)
            a=np.linalg.solve(A,b)
    
            C[i][j]=(-32./pi)*sum([b[i]*a[i] for i in range(len(a))])    
            
    M = np.linspace(0, m, m+1)
    N = np.linspace(1, n, n+1)    
    
    return M,N,C
    

def u(m,n, delitev): 
    
    A=Aij(m,n)
    b=bj(m,n)
    a=np.linalg.solve(A,b)
    
    dphi = pi/2/delitev
    dr = 1./delitev
    U=np.zeros((delitev+1,2*delitev+1))
    
    for i in range(delitev+1):
        r = dr*i
        print(r)
        for j in range(2*delitev+1):
            phi = dphi*j
            
            Phi=Phimn(r,phi,m,n)
            U[i][j]=sum([Phi[k]*a[k] for k in range(len(a))])


    Rad = np.linspace(0, 1., delitev+1)
    PHI = np.linspace(0, pi, 2*delitev+1)
    
    return Rad,PHI,U
    
# ***********************    IZRAČUNI   ****************************
m=30  #št. členov v vsoti C po m,s
n=30  #št. členov v vsoti C po m,s

# izračun za C
M,N,koeficient = C(m,n)
#koeficient(m=50,n=50)=0.75772062981184074
#koeficient(m=100,n=100)=0.7577218721911726


   
## izračun hitrosti
delitev=200

Rad,PHI,hitrost = u(m,n, delitev)


# ***********************    GERAFIRANJE   ****************************

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib import colors, ticker, cm

import matplotlib.ticker as mticker
def log_tick_formatter(val, pos=None):
    return "{:.0e}".format(10**val)


P, R = np.meshgrid(PHI,Rad)
# transform them to cartesian system
XX, YY = R*np.cos(P), R*np.sin(P)

fig = plt.figure()
#ax1 = fig.gca(projection='3d')
#surf = ax1.plot_surface(XX,YY,hitrost, rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0)
#cs = plt.contourf(XX, YY, hitrost,  cmap=cm.PuBu_r)
ax1 = fig.add_subplot(111)
cs = plt.contourf(XX, YY, hitrost,  cmap=cm.coolwarm)
#ax2 = fig.add_subplot(222)
#cs = plt.contourf(XX, YY, hitrost2,  cmap=cm.coolwarm)
#ax3 = fig.add_subplot(223)
#cs = plt.contourf(XX, YY, hitrost3,  cmap=cm.coolwarm)
#ax4 = fig.add_subplot(224)
#cs = plt.contourf(XX, YY, hitrost4,  cmap=cm.coolwarm)

#ax1 = fig.add_subplot(311)
#cs = plt.contourf(XX, YY, abs(hitrost4-hitrost1), locator=ticker.LogLocator(), cmap=cm.coolwarm)
#ax2 = fig.add_subplot(312)
#cs = plt.contourf(XX, YY, abs(hitrost4-hitrost2), locator=ticker.LogLocator(), cmap=cm.coolwarm)
#ax3 = fig.add_subplot(313)
#cs = plt.contourf(XX, YY, abs(hitrost4-hitrost3), locator=ticker.LogLocator(), cmap=cm.coolwarm)

#ax1 = fig.add_subplot(311)
#cs = plt.contourf(XX, YY, (hitrost4-hitrost1), cmap=cm.coolwarm)
#ax2 = fig.add_subplot(312)
#cs = plt.contourf(XX, YY, (hitrost4-hitrost2), cmap=cm.coolwarm)
#ax3 = fig.add_subplot(313)
#cs = plt.contourf(XX, YY, (hitrost4-hitrost3), cmap=cm.coolwarm)
#
cbar = plt.colorbar()
#ax1.set_title( 'Tridimenzionalen prikaz hitrostnega profila ($n=$'+str(n)+',$m=$'+str(m)+')')
ax1.set_title( 'Prikaz hitrostnega profila ($n=$'+str(n)+',$m=$'+str(m)+')')

#ax1.set_title( '$n=$'+str(n1))
#ax2.set_title( '$n=$'+str(n2))
#ax3.set_title( '$n=$'+str(n3))
#ax4.set_title( '$n=$'+str(n4))

#ax1.set_title( '$|U_{n=3}-U_{n=20}|$')
#ax2.set_title( '$|U_{n=5}-U_{n=20}|$')
#ax3.set_title( '$|U_{n=10}-U_{n=20}|$')
#
#ax1.set_title( '$U_{n=3}-U_{n=20}$')
#ax2.set_title( '$U_{n=5}-U_{n=20}$')
#ax3.set_title( '$U_{n=10}-U_{n=20}$')

ax1.set_xlabel("x")
ax1.set_ylabel("y")
#ax2.set_xlabel("x")
#ax2.set_ylabel("y")
#ax3.set_xlabel("x")
#ax3.set_ylabel("y")
#ax4.set_xlabel("x")
#ax4.set_ylabel("$|U_{n}-U_{n=20}|$")
#ax1.set_zlabel("u(x,y)")
#ax1.set_zlim([0,0.1])
#ax1.legend(loc='best')
plt.tight_layout()
plt.show()



MM, NN = np.meshgrid(M,N)
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')

surf = ax2.plot_surface(MM,NN,koeficient, rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0)
ax2.set_title( 'Izračuni koeficienta $C_{m,n}$ za različne kombinacije vrednosti $n$ in $m$')

#surf = ax2.plot_surface(MM,NN,np.log10(abs(0.757721-koeficient)), rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0)
#ax2.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
#ax2.set_title( 'Odstopanja izračunov koeficientov $C_{m,n}$ od rešitve $C_{m=n=100} = 0.757721$')

plt.xlabel('m')
plt.ylabel('n')
ax2.set_zlabel("C(m,n)")
#ax2.set_zlabel("$|C(m,n)-C_{100}|$")
#ax2.set_zlim([0,0.1])
#plt.zscale('log')
#ax2.zaxis.set_scale('log')
ax2.legend(loc='best')
plt.tight_layout()
plt.show()

