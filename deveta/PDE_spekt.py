# -*- coding: utf-8 -*-

import scipy as sc
import numpy  as np
import matplotlib.pyplot as plt
from numpy import linalg

#*************************************************
pi=3.1415926535897932384626433

L=1.     # debelina plasti 
N=400   # delitev plasti 0.5/500 = 0.001  -  frekvenca vzorčenja= 1000  -  Nyquistova frekvenca = 500
D=2.6*10**(-4)     # difuzijska konstanta smrekov les  lamb=0.16 [W/mK] rho=450 [kg/m3] c=1360[J/kgK]
sig=0.2  # gaussov sigma porazdelitve temperature po plasti
#cas=400004# časovni interval!
cas=200002# časovni interval!
#cas=101# časovni interval!

#*************************************************

def Difuzijska(N,L,D,sig,cas):
    
    Txt0=[]
    
    for j in range(N+1): #začetni pogoj
        
        Txt0.append(sc.exp(-(j*L/N -L/2.)**2/(sig**2.))) # vektor gaussove porazdelitev temperature po plasti

    Tkfft=np.zeros((cas,N+1)) # matrika časovnih korakov razdelejene plasti v fourireovem prostoru
    Tkfft[0,:]=sc.fft(Txt0) # v matriko časovnih korakov vpišem začetni približek

#    ht=10**(-2) # št. časovnih korakov po časovnem intervalu cas
    ht=10**(-3) # št. časovnih korakov po časovnem intervalu cas
#    ht=10**(-3) # št. časovnih korakov po časovnem intervalu cas

    for j in range(1,cas-1,1):   # Eulerjev razvoj fft približka Txt0 v časovnem intervalu (po vseh Tkfft)
        for k in range(N+1):
            
            Tkfft[j,k]=Tkfft[j-1,k]-ht*D*(2.*k*pi/L)**2.*Tkfft[j-1,k] # Eulerjev razvoj fft skozi čas
            
            if (Tkfft[j,k]/Tkfft[j-1,k])>1.0: # pogoj za stabilnost Eulerjeve diferenčne sheme                
                print("nestabilnost")
                return 
                
    Txifft=sc.ifft(Tkfft) # matrika časovnih korakov razdelejene plasti preslikamo iz fourireovega prostora
#    Txifft=np.zeros((cas,N+1)) #        
#    Rez=np.matrix([[0.1*sc.sin(j*10.0/N)+0.1 for j in range(N+1)] for k in range(cas)])
#         
#    for j in range(0,cas-1,1):
#        
#        Txifft[j:j+1]=sc.ifft(Tkfft[j:j+1])
#        
#        for k in range(N+1):
#            if Txifft[j,k]>2: #Nekaj pozornosti zahteva tudi diskretizacija: za vsak k mora biti 0< (N-1)/L< fNyquist = N/2L
#                return Rez
#                
    return Txifft
    
#********************** izračuni ***************************
M=Difuzijska(N,L,D,sig,cas)

X = np.linspace(0,L,N+1) 
T = np.linspace(0,cas,cas) 

INT=[]
for j in range(0,cas-1,1):
    INT.append(np.linalg.norm(M[j:j+1]))

#********************** grafiranje ***************************
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
##
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#XX,TT = np.meshgrid(X,T)
##
##surf = ax.plot_wireframe(TT,XX,M)
#surf = ax.plot_surface(TT,XX,M , cmap=cm.coolwarm, linewidth=0.01, antialiased=False)
##ax.plot_surface(TT, XX, M, rstride=8, cstride=8, alpha=0.3)
##cset = ax.contourf(TT, XX, M, zdir='m', offset=-100, cmap=cm.coolwarm)
##cset = ax.contourf(TT, XX, M, zdir='t', offset=-40, cmap=cm.coolwarm)
##cset = ax.contourf(TT, XX, M, zdir='y', offset=40, cmap=cm.coolwarm)
#ax.set_title( 'Difuzija začetnega temperaturnega profila skozi čas')
#ax.set_xlabel("$h_{t}$")
#ax.set_ylabel("L")
#ax.set_zlabel("T(L,t)")
#plt.tight_layout()
#plt.show()

#M0=np.array([row[1] for row in M])
#M1=np.array([row[10] for row in M])
#M2=np.array([row[100] for row in M])
#M3=np.array([row[1000] for row in M])
#M4=np.array([row[10000] for row in M])
#

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(X,M[10,:],"k",label='t=0.001')
ax1.plot(X,M[100,:],"k--",label='t=0.1')
ax1.plot(X,M[100000,:],"k-.",label='t=100')
ax1.plot(X,M[200000,:],"k:",label='t=200')
#ax1.plot(X,M[400000,:],"k:",label='t=200') 
ax1.legend(loc='upper right')
ax1.set_title( 'Difuzija začetnega temperaturnega profila skozi čas')
plt.xlabel('L')
plt.ylabel('Temperatura')
plt.show()


tt = np.linspace(0,cas-1,cas-1) 
#
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
ax2.plot(tt,INT/INT[0],"k:",label='N=400'))
#ax2.plot(tt,INT/INT[0],"k--,label='N=100')")
#ax2.plot(tt,INT/INT[0],"k",label='N=50'))
ax2.set_title( 'Integral temperature celotnega temperaturnega profila v časovnih korakih')
plt.xlabel('$h_{t}$')
plt.ylabel('Vsota temperature')
plt.xlim([0, cas]) 
plt.show()