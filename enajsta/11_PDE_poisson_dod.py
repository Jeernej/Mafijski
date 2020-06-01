# -*- coding: utf-8 -*-
import scipy as sc
import numpy  as np
from scipy.special import *
from math import *

# *************************************************
pi=3.1415926535897932384626433

# izračun koeficienta C za pravokotni profil
def C(a,b,eps):    

    vrsta=0
    V=[]
 
    n=1
    popravek=1
    while popravek>eps:
        popravek=np.tanh((2*n-1)*a*0.5*pi/b)/((2*n-1)**5)
        vrsta=vrsta+popravek
        n=n+1
        V.append(vrsta)
#        print(popravek)
        
    koef=2.*pi*b/a*(1./3-b*64./(a*pi**5)*vrsta)
    V=2.*np.pi*b/a*(1./3-b*64./(a*np.pi**5)*np.array(V))
    return koef,V


# *************************************************
# izračun hitrosti za kvadratni profil a=b po lastnih funkcijah: d_n cosh((2n-1)*pi*x/b) * cos((2n-1)\pi*y/b)

def gmn1a(y,n,b): #partikularna rešitev
    return sc.cos((2.*n-1)*pi*y/b)
        
def Bn(n,b):  # nastavek za izračun uteži - množim še z eta/(p'*b^2) za brezdimenzijsko obliko    
    bn=2.*((2.*n-1)*pi*np.sin(n*pi)+2.*np.cos(n*pi))/((2.*n-1)*pi)**3.
    return bn

def gmn1b(x,y,n,b): #homogena rešitev
    return  sc.cosh((2.*n-1)*pi*x/b)*sc.cos((2.*n-1)*pi*y/b)
    
def Dn(n,b):  # nastavek za izračun uteži - množim še z eta/(p'*b^2) za brezdimenzijsko obliko    
    dn= -1.*Bn(n,b)/sc.cosh((2.*n-1)*pi)
    return dn
    
#v_p(x,y)=\frac{p'}{2 \eta}y^2+c_1y+c_2=\sum_n b_n \cos((2n-1)\pi y/b) \  .   
# 
def u1(nMax,delitev,b):

    dy = b/delitev
    dx = b/delitev

    X=[]
    Y=[]
    U=np.zeros((delitev+1,delitev+1))

    for i in range(delitev+1):
        x = dx*i
        print(x)
        for j in range(delitev+1):
            y = dy*j
            
            U[i][j]=sum(Bn(n,b)*gmn1a(y,n,b) for n in range(1,nMax,2))   #partikularna rešitev
#            U[i][j]=sum(Dn(n,b)*gmn1b(x,y,n,b) for n in range(1,nMax,2))  #homogena rešitev

            X.append(x)
            Y.append(y)
            
    return X,Y,U
  
  
  
# *************************************************
# izračun hitrosti po lastnih funkcijah pravokotnika: sin(n*pi*x/a)*sc.sin(m*pi*y/b)

def gmn2(x,y,m,n,a,b):
    return sc.sin(n*pi*x/a)*sc.sin(m*pi*y/b)
    
def Amn2(m,n,a,b):  # nastavek za izračun uteži po lastnih funkcijah pravokotnika    
    if m%2==1 and n%2==1: return 16.0/(m*n*pi**4)/((n/a)**2+(m/b)**2)
    else: return 0
    
def u2(mMax,nMax,delitev,a,b):

    dy = b/delitev
    dx = a/delitev

    X=[]
    Y=[]
    U=np.zeros((delitev+1,delitev+1))
    
    for i in range(delitev+1):
        x = dx*i
        print(x)
        for j in range(delitev+1):
            y = dy*j
            
            U[i][j]=sum(Amn2(m,n,a,b)*gmn2(x,y,m,n,a,b) for m in range(1,mMax,2) for n in range(1,nMax,2))

            X.append(x)
            Y.append(y)
            
    return X,Y,U



#def grafP(m,n,a,b,tocke):
#    zapis=open("grafP"+str(m)+str(n)+".txt","w")
#
#    dy = 1.0*b/tocke
#    dx = 1.0*a/tocke
#    for i in range(tocke+1):
#        x = dx*i
#        for j in range(tocke+1):
#            y = dy*j
#            zapis.write("{: > 010,.08e}  {: > 010,.08e}  {: > 030,.22f}\n".format(x,y,Gmn(x,y,m,n,a,b)))
#    zapis.close()   
    
    
    #********************** IZRAČUNI ***************************

n=80   # št. členov v vsoti C po m,s
delitev1=50
delitev=200
eps=10**(-15)
a=1.
b=0.1

#### izračun za C
#koeficient,V = C(a,b,eps)
#
#konvergenca = [abs(V[m-1]-V[m]) for m in range(len(V)-1)]  

#raz=[]
#CC=[]
#for i in range(1,2000):
#        koeficient,V = C(a*i/200,b,eps)
#        CC.append(koeficient)
#        raz.append((a*i/200)/b)
## izračun hitrosti
#X,Y,hitrost1 = u1(n,delitev,b)

X,Y,hitrost1 = u2(n,n,delitev1,a,b)
#X,Y,hitrost2 = u2(n,n,delitev,a,b)


#********************** GRAFIRANJE ***************************
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

#X = np.linspace(0, a, delitev+1)
#Y = np.linspace(0, b, delitev+1)
X = np.linspace(0, a, delitev1+1)
Y = np.linspace(0, b, delitev1+1)
XX,YY = np.meshgrid(X,Y)

fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(XX,YY,hitrost1, rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0)
#surf = ax.plot_surface(XX,YY,hitrost2, rstride=10, cstride=1, cmap=cm.coolwarm,linewidth=0)
#surf = ax.plot_surface(XX,YY,hitrost1-hitrost2, rstride=10, cstride=0, cmap=cm.coolwarm)
#surf = ax.plot_surface(XX,YY,abs(hitrost1-hitrost2), locator=ticker.LogLocator(), rstride=10, cstride=0, cmap=cm.coolwarm)
ax = fig.add_subplot(111)
cs = plt.contourf(XX, YY, hitrost1,  cmap=cm.coolwarm)
#cs = plt.contourf(XX, YY, hitrost2,  cmap=cm.coolwarm)

#cbar = plt.colorbar()
ax.set_title( 'Hitrostni profil kvadratne cevi')
ax.set_xlabel("x")
ax.set_ylabel("y")
#ax.set_zlabel("u(x,y)")
#ax.set_ylim([-20,20])
#plt.zscale('log')
ax.legend(loc='best')
plt.tight_layout()
plt.show()

#
t =  np.linspace(0,len(V),len(V))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(t,V,"k")#,label='t=0.001')
#ax2.legend(loc='lower right')
ax2.set_title( 'Konvergenca koeficienta k rešitvi $C=$'+str(koeficient)+' pri korakih vsote ($N_{h_C}=$'+str(len(konvergenca))+')')
plt.xlabel('$h_C$')
plt.ylabel('C')
plt.xlim([0,len(V)])
#plt.yscale('log')
plt.show()


T = np.linspace(0,len(konvergenca),len(konvergenca)) 
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(T,konvergenca,"k--")#,label='t=0.001')
#ax2.legend(loc='lower right')
ax1.set_title( 'Odstopanja vrednosti koeficienta $C=$'+str(koeficient)+' med zaporednimi koraki ${h_C}$ v vsoti')
plt.xlabel('$h_C$')
plt.ylabel('$|C_{n+1}-C_{n}|$')
plt.xlim([0,len(konvergenca)])
plt.yscale('log')
plt.show()


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(raz,CC,"k")#,label='t=0.001')
#ax2.legend(loc='lower right')
ax2.set_title( 'Vrednosti koeficienta $C$ pri različnih razmerjih stranic profila cevi')# ($N_{h_C}=$'+str(len(konvergenca))+')')
plt.xlabel('$a/b$')
plt.ylabel('C')
#plt.xlim([0,len(V)])
#plt.yscale('log')
plt.show()