# -*- coding: utf-8 -*-
import scipy as sc
import numpy  as np
from scipy.special import *
from math import *

# *************************************************
pi=3.1415926535897932384626433

#  Besselove funkcije
#def rekurzija(x,N):
#    """vrne seznam Besselovih funkcij od 0-te do N-te"""
#    Jn0 = 10**-6 
#    Jn1 = 0
#    Jnji = [Jn0,Jn1]
#
#    for i in range(2*N,0,-1):
#        Jn_1 = 2.0*i*Jn0/x - Jn1
#        Jnji[:0] = [Jn_1]
#        Jn1,Jn0 = Jn0,Jn_1
#   
#    norma = Jnji[0] + 2*sum(Jnji[2*i] for i in range(1,len(Jnji)/2))
#
#    for i in range(len(Jnji)):
#        Jnji[i]=Jnji[i]/norma
#    return Jnji

#def Nicle(x,N):
  
# izračun Ims(m,s) z rekurzijo
#def Ims(m,s,N):
#    koef = 4*(2*m+1)/sc.special.jn_zeros(2*m+1,s)[s-1]
#    vsota = 0
#    for k in range(m+1,int(N)):
#        vsota += k*rekurzija(sc.special.jn_zeros(2*m+1,s)[s-1],N)[2*k]/(4*k**2 - 1)
#    return koef*vsota
    
# izračun koeficienta Ams(m,s) z rekurzijo
#def Ams(m,s,N):
#    stevec = 8.0* Ims(m,s,N)
#    imenovalec = (2*m+1)*pi*sc.special.jn(2*m+2,sc.special.jn_zeros(2*m+1,s)[s-1])**2  #ničla = jn_zeros(n,zap.nicla)
#    return stevec/imenovalec

# izračun koeficienta pretoka za polkrožni profil    
#def C(n):
#
#    vsota = 0
#    
#    for m in range(0,n+1):
##        print(m)
#        for s in range(1,n+2):
#            im = pi*(2*m+1)*sc.special.jn_zeros(2*m+1,s)[s-1]*sc.special.jn(2*m+2,sc.special.jn_zeros(2*m+1,s)[s-1])
#            st = 8*Ims(m,s,n)
#            vsota+= (st/im)**2
#            
#    return 8*vsota

# grafi lastnih funkcij polkroga
#def graf(m,s,N,tocke):
#    zapis=open("graf"+str(m)+str(s)+".txt","w")
#    koef = Ams(m,s,N)
#    dfi = pi/tocke
#    dx = 1.0/tocke
#    for i in range(tocke+1):
#        x = dx*i
#        for j in range(tocke+1):
#            fi = dfi*j
#            vr = sc.special.jn(2*m+1,sc.special.jn_zeros(2*m+1,s)[s-1]*x)*sc.sin((2*m+1)*fi)
#            zapis.write("{: > 010,.08e}  {: > 010,.08e}  {: > 030,.22f}\n".format(x*sc.cos(fi),x*sc.sin(fi),vr))
#    zapis.close()            


# izračun vrednosti lastnih funkcij, ničel, koeficienta Ams(m,s)
# in vrednoti Ims(m,s) s funkcijami iz modula scipy.special

def nicla(m,s): 
    return sc.special.jn_zeros(2*m+1,s)[s-1] # poišče s-to ničlo besselove funkcije reda 2m+1

def gms(x,phi,m,s):  # vrednost lastne funkcije polkroga g_{ms}(x, phi)  v točki x,phi
    return sc.special.jn(2*m+1,nicla(m,s)*x)*sc.sin((2*m+1)*phi)
    
def ImsPy(m,s,N): # vrste ne seštevam od m+1 do neskončno ampak le do m+N+2
    vsota = (4*(2*m+1)/nicla(m,s)) * sum(k*sc.special.jn(2.*k,nicla(m,s))/(4.*k**2-1.) for k in range(m+1,m+N+2))     
    return vsota  #dolgotrajno računanje
    
# izračun koeficienta C   
def C(n,N): 
    vsota = 0  
    konv = [0]
    for m in range(0,n+1):        
        konv[m]=vsota
        print(m)
        for s in range(1,n+2):  # zamaknjena za eno mesto od m ..
            A = 8*ImsPy(m,s,N)
            B = pi*(2*m+1)*nicla(m,s)*sc.special.jn(2*m+2,nicla(m,s))
            vsota+= (A/B)**2           
        konv.append(abs(konv[m]-vsota))
        print(vsota)    
        print(konv)    

    return konv


# izračun hitrosti za polkrožni profil 
def AmsPy(m,s,N):  #dolgotrajno računanje
    return 8*ImsPy(m,s,N) / ((2*m+1)*pi*sc.special.jn(2*m+2,nicla(m,s))**2)
    
def u(mMax,sMax,delitev): 

    dphi = pi/2/delitev
    dr = 1./delitev

    X=[]
    Y=[]
    U=np.zeros((delitev+1,2*delitev+1))
    
    Ams=[[AmsPy(m,s,N) for s in range(1,sMax)] for m in range(0,mMax)]   #dolgotrajno računanje
    
    for i in range(delitev+1):
        r = dr*i
        print(r)
        for j in range(2*delitev+1):
            phi = dphi*j
            U[i][j]=sum(Ams[m][s-1]*gms(r,phi,m,s)/nicla(m,s)**2 for m in range(0,mMax) for s in range(1,sMax-1))
#            X.append(r*sc.cos(phi))
#            Y.append(r*sc.sin(phi))
            
    return X,Y,U


## graf vsote za polkrožni profil       
#def graf_vsota(mMax,sMax,delitev):
#    zapis=open("vsota"+str(mMax)+str(sMax)+".txt","w")
#    dfi = pi/delitev
#    dr = 1.0/delitev
#
#    amji=[[AmsPy(m,s) for s in range(1,sMax)] for m in range(0,mMax)]
#    print(amji)
#    for i in range(delitev+1):
#        r = dr*i
#        for j in range(delitev+1):
#            fi = dfi*j
#            vs=sum(amji[m][s-1]*gms(r,fi,m,s) for m in range(0,mMax) for s in range(1,sMax-1))
#            zapis.write("{: > 010,.08e}  {: > 010,.08e}  {: > 030,.22f}\n".format(r*sc.cos(fi),r*sc.sin(fi),vs))
#    zapis.close()

#********************** IZRAČUNI ***************************

N=190  # št. členov v vsoti za računanje ImsPy
n1=3  # št. členov v vsoti C po m,s
n2=5  # št. členov v vsoti C po m,s
n3=10  # št. členov v vsoti C po m,s
n4=20  # št. členov v vsoti C po m,s

delitev=200
#7.57721437e-01,
#         9.29279564e-09
## izračun za C
#konvergenca = C(n,N)
##
#koeficient=konvergenca
#konvergenca = [abs(koeficient[m-1]-koeficient[m]) for m in range(len(koeficient)-1)]  

## izračun hitrosti
X,Y,hitrost1 = u(n1,n1,delitev)
X,Y,hitrost2 = u(n2,n2,delitev)
X,Y,hitrost3 = u(n3,n3,delitev)
X,Y,hitrost4 = u(n4,n4,delitev)

#********************** GRAFIRANJE ***************************
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib import colors, ticker, cm

Rad = np.linspace(0, 1., delitev+1)
PHI = np.linspace(0, pi, 2*delitev+1)

P, R = np.meshgrid(PHI,Rad)
# transform them to cartesian system
XX, YY = R*np.cos(P), R*np.sin(P)

fig = plt.figure()
ax1 = fig.gca(projection='3d')
surf = ax1.plot_surface(XX,YY,hitrost, rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0)
#cs = plt.contour(XX, YY, hitrost,  cmap=cm.PuBu_r)
#ax1 = fig.add_subplot(221)
#cs = plt.contourf(XX, YY, hitrost1,  cmap=cm.coolwarm)
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
#cbar = plt.colorbar()
ax1.set_title( 'Tridimenzionalen prikaz hitrostnega profila ($n=$'+str(n4)+')')
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
ax1.set_zlabel("u(x,y)")
ax1.set_zlim([0,0.1])
ax1.legend(loc='best')
plt.tight_layout()
plt.show()

#
#t =  np.linspace(0,n+1,n+2)
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.plot(t,8.*np.array(koeficient),"k")#,label='t=0.001')
##ax2.legend(loc='lower right')
#ax2.set_title( 'Konvergenca koeficienta k rešitvi $C=$'+str(8*koeficient[n])+' pri korakih vsote ($N_{h_C}=$'+str(len(konvergenca))+')')
#plt.xlabel('$h_C$')
#plt.ylabel('C')
#plt.xlim([0,80])
##plt.yscale('log')
#plt.show()
#
#
#
#T = np.linspace(0,n,n+1) 
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.plot(T,8.*np.array(konvergenca),"k--")#,label='t=0.001')
##ax2.legend(loc='lower right')
#ax1.set_title( 'Odstopanja vrednosti koeficienta $C=$'+str(8*koeficient[n])+' med zaporednimi koraki ${h_C}$ v vsoti')
#plt.xlabel('$h_C$')
#plt.ylabel('$|C_{n+1}-C_{n}|$')
#plt.xlim([0,80])
#plt.yscale('log')
#plt.show()

