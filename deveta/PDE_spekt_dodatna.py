import scipy as sc
import numpy  as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#from scipy import special
from numpy import linalg

#*************************************************

pi=3.1415926535897932384626433
L=1.     # debelina plasti 
N=100    # delitev plasti 0.5/500 = 0.001  -  frekvenca vzorčenja= 1000  -  Nyquistova frekvenca = 500
D=2.6*10**(-4)     # difuzijska konstanta smrekov les  lamb=0.16 [W/mK] rho=450 [kg/m3] c=1360[J/kgK]
sig=0.2  # gaussov sigma porazdelitve temperature po plasti
cas=100004# časovni interval!
#cas=400004# časovni interval!
##cas=200002# časovni interval!
##cas=100001# časovni interval!
ht=10**(-2) # dolžina koraka na časovnem intervalu!

#*************************************************

# KOLOKACIJSKA METODA
# N = stevilo podintervalov
#def TzacKol(x,L,T0=1.0,sigma=0.2):
#    return T0*e**(-(x-L/2)**2 / sigma**2)

def priblizek(L,N,sig=0.2):   
#    G = [6*TzacKol(i*L/N,L) for i in range(1,N)]
#    G = [6*sc.exp(-(i*L/N-L/2)**2 / sig**2) for i in range(1,N-1)]  # začetni gaussova porazdelitev
    G = [6*sc.sin(j*pi/N) for j in range(1,N-1)]  # začetni sinusna porazdelitev
    return G
    
def Zlepki(cas,ht,N,L,D,nacin):

    dx = L/N
    dX = np.linspace(0,L,N+1)   # delitev intervala L na N+1 točk 
    
    A = 4*np.identity(N-2)    # ustvarimo matriki A in B
    koef = 6*D/dx**2
    B = -2*koef*np.identity(N-2)
    for i in range(0,N-3):
        A[i][i+1],A[i+1][i] = 1,1
        B[i][i+1],B[i+1][i] = koef,koef  # ustvarimo matriki A in B

    Tk=np.zeros((cas,N)) # matrika časovnih korakov razdelejene plasti 
    Tk[0,1:N-1]=priblizek(L,N) # v matriko časovnih korakov vpišem začetni približek (robova intervala sta 0 ob vsakem času)

    a0 = np.dot(np.linalg.inv(np.array(A)),priblizek(L,N)) # začetni približek za kolokacijsko metodo
    for i in range(1,cas): # preskočim prvi korak na časovnem intervalu! - tam je že začetni približek
        
        if nacin == "EE":
            an1 = ekspEuler(a0,ht,A,B,N)
        elif nacin == "IE":
            an1 = impEuler(a0,ht,A,B)
            
        an2 = an1
        an2 = np.insert(an2,0,0)  # robmi pogoji pri x=0 za izračun vsote tep. profila
        an2 = np.insert(an2,0,-an1[0]) # robmi pogoji pri x=0 za izračun vsote tep. profila
        an2 = np.append(an2,[0,-an2[len(an2)-1]]) # robmi pogoji pri x=N  za izračun vsote tep. profila
        for j in range(0,N-1):
            Tk[i,j]=T(j*dx,an2,dX,dx,N) # izračun vsote tep. profila (PREVERI: robova intervala morata biti 0 ob vsakem času)
            
        a0 = an1   
        
    return Tk
         

# reševanje matričnega sistema v Aa'=Ba skozi čas 
def ekspEuler(an,ht,A,B,N):
    Desna = np.identity(N-2) + ht*np.dot(np.linalg.inv(np.array(A)),np.array(B))
    an1=np.dot(Desna,an)
    return an1

def impEuler(an,ht,A,B):
    Leva = np.array(A) - ht/2.0*np.array(B)
    Desna = np.array(A) + ht/2.0*np.array(B)
    an1=np.linalg.solve(Leva,np.dot(Desna,an))
    return an1

def T(x,an2,DD,dx,N):
    Tz = 0.
    for i in range(0,N+2):
        k=i-1
        Tz += an2[k+1]*Bzlepki(k,x,DD,dx)
    return Tz

def Bzlepki(k,x,DD,dx):
#    n = len(DD)
    if x <= DD[k-2]: 
        return 0
    elif DD[k-2] <= x <= DD[k-1]:
        rez = 1.0/(6*dx**3)*(x-DD[k-2])**3
        return rez
    elif DD[k-1] <= x <= DD[k]:
        rez = 1.0/6 + 1.0/(2*dx)*(x-DD[k-1]) + 1.0/(2*dx**2)*(x-DD[k-1])**2 - 1.0/(2*dx**3)*(x-DD[k-1])**3
        return rez
    elif DD[k] <= x <= DD[k+1]:
        rez = 1.0/6 - 1.0/(2*dx)*(x-DD[k+1]) + 1.0/(2*dx**2)*(x-DD[k+1])**2 + 1.0/(2*dx**3)*(x-DD[k+1])**3
        return rez    
    elif DD[k+1] <= x <= DD[k+2]:
        rez = -1.0/(6*dx**3)*(x-DD[k+2])**3
        return rez        
    elif x >= DD[k+2]: 
        return 0    
    
# Frekvenčni spekter začetnega temperaturnega profila
#def spekter(N,sigma,T0=1.0,L=1.0):
#    f = open("spekter.txt","w")
#    sp=fourier(N,T0,L,sigma)
#    for i in range(len(sp)):
#        ss = sp[i].real**2+sp[i].imag**2
#        f.write("{: > 010,.02e}  {: > 010,.14e}\n".format(i,ss))
#    f.close()

#********************** izračuni ***************************
Me=Zlepki(cas,ht,N,L,D,nacin='EE')
#Mi=Zlepki(cas,ht,N,L,D,nacin='IE')

X = np.linspace(0,L,N) 
T = np.linspace(0,cas,cas) 

INT=[]
for j in range(0,cas-1,1):
    INT.append(np.linalg.norm(M[j:j+1]))

#********************** grafiranje ***************************
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
#
#
EKSfig = plt.figure()
EKSax = EKSfig.gca(projection='3d')
XX,TT = np.meshgrid(X,T)
#
#surf = ax.plot_wireframe(TT,XX,M)
surf = EKSax.plot_surface(TT[1:10000,:],XX[1:10000,:],Me[1:10000,:] , cmap=cm.coolwarm, linewidth=0.0, antialiased=False)
#ax.plot_surface(TT, XX, M, rstride=8, cstride=8, alpha=0.3)
#cset = ax.contourf(TT, XX, M, zdir='m', offset=-100, cmap=cm.coolwarm)
#cset = ax.contourf(TT, XX, M, zdir='t', offset=-40, cmap=cm.coolwarm)
#cset = ax.contourf(TT, XX, M, zdir='y', offset=40, cmap=cm.coolwarm)
EKSax.set_title( 'Difuzija začetnega temperaturnega profila skozi čas (eksplicitno)')
EKSax.set_xlabel("$h_{t}$")
EKSax.set_ylabel("L")
EKSax.set_zlabel("T(L,t)")
plt.tight_layout()
plt.show()
#

EKSfig1 = plt.figure()
EKSax1 = EKSfig1.add_subplot(111)
EKSax1.plot(X,Me[10,:],"k",label='t=0.1')
#EKSax1.plot(T,Me[:,3],"k",label='t=0.1')
EKSax1.plot(X,Me[1000,:],"k--",label='t=10')
EKSax1.plot(X,Me[10000,:],"k-.",label='t=100')
EKSax1.plot(X,Me[100000,:],"k:",label='t=1000')
#EKSax1.plot(X,Me[200000,:],"k-.",label='t=100')
#aEKSx1.plot(X,Me[400000,:],"k:",label='t=200') 
EKSax1.legend(loc='upper right')
EKSax1.set_title( 'Difuzija začetnega temperaturnega profila skozi čas (eksplicitno)')
plt.xlabel('L')
plt.ylabel('Temperatura')
plt.ylim([0, 1]) 
plt.show()


#fig = plt.figure()
#ax = fig.gca(projection='3d')
#XX,TT = np.meshgrid(X,T)
##
##surf = ax.plot_wireframe(TT,XX,M)
#surf = ax.plot_surface(TT[1:1000,:],XX[1:1000,:],Mi[1:1000,:] , cmap=cm.coolwarm, linewidth=0.01, antialiased=False)
##ax.plot_surface(TT, XX, M, rstride=8, cstride=8, alpha=0.3)
##cset = ax.contourf(TT, XX, M, zdir='m', offset=-100, cmap=cm.coolwarm)
##cset = ax.contourf(TT, XX, M, zdir='t', offset=-40, cmap=cm.coolwarm)
##cset = ax.contourf(TT, XX, M, zdir='y', offset=40, cmap=cm.coolwarm)
#ax.set_title( 'Difuzija začetnega temperaturnega profila skozi čas (implicitno)')
#ax.set_xlabel("$h_{t}$")
#ax.set_ylabel("L")
#ax.set_zlabel("T(L,t)")
#plt.tight_layout()
#plt.show()
#
#
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.plot(X,Mi[1,:],"k",label='t=0.1')
#ax1.plot(X,Mi[100,:],"k--",label='t=10')
#ax1.plot(X,Mi[1000,:],"k-.",label='t=100')
#ax1.plot(X,Mi[10000,:],"k:",label='t=1000')
##ax1.plot(X,Mi[200000,:],"k-.",label='t=100')
##ax1.plot(X,Mi[400000,:],"k:",label='t=200') 
#ax1.legend(loc='upper right')
#ax1.set_title( 'Difuzija začetnega temperaturnega profila skozi čas (implicitno)')
#plt.xlabel('L')
#plt.ylabel('Temperatura')
#plt.show()
#

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(X,abs(Me[10,:]-Mi[1,:]),"k",label='t=0.1')
ax1.plot(X,abs(Me[1000,:]-Mi[100,:]),"k--",label='t=10')
ax1.plot(X,abs(Me[10000,:]-Mi[1000,:]),"k-.",label='t=100')
ax1.plot(X,abs(Me[100000,:]-Mi[10000,:]),"k:",label='t=1000')
#ax1.plot(X,Mi[200000,:],"k-.",label='t=100')
#ax1.plot(X,Mi[400000,:],"k:",label='t=200') 
ax1.set_yscale('log')
ax1.legend(loc='upper left')
ax1.set_title( 'Difuzija začetnega temperaturnega profila skozi čas (implicitno)')
plt.xlabel('L')
plt.ylabel( '$|T_{im}-T_{ek}|$' )
ax1.set_title( 'Odstopanje med implicitnim in eksplicitnim izračunom difuzije začetnega temperaturnega profila skozi čas')
plt.show()


#
#tt = np.linspace(0,cas-1,cas-1) 
#
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.plot(tt,INT,"k--")
#ax2.set_title( 'Integral temperature celotnega temperaturnega profila v časovnih korakih')
#plt.xlabel('$h_{t}$')
#plt.ylabel('Vsota temperature')
#plt.ylim([0, 5]) 
#plt.xlim([0, cas]) 
#plt.show()

