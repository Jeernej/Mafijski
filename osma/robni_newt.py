# -*- coding: utf-8 -*-
import scipy as sc
import numpy  as np
import matplotlib.pyplot as plt

#*************************************************

N=50

a=0
b=1

h=(b-a)*1.0/N  # koraki med točkami

#delta=0.5
#ksi1=0.2583923655030188
#ksi2=3.2595598244395485

#delta=1.
#ksi1=0.379291149762689
#ksi2=2.734675693030527

#delta=3.0
#ksi1=0.8433769410714723
#ksi2=1.6441423148135939

#delta=3.5
#ksi1=1.1379634157095866
#ksi2=1.2635856746592544


#*************************************************

def ddy(y,delta):    #G-B enačba
    return (-delta)*sc.exp(y)

def Analiticna(N,ksi): #analitična rešitev enačbe G-B

    a=0
    b=1
    h=(b-a)*1.0/N 
    A=[]
    
    for j in range(0,N+1):
        
        A.append(-2*sc.log( sc.cosh( ksi*(1-2.0*j*h))/sc.cosh(ksi) ) )
        
    return A


def pribl(mu,N):     # konveksni začetni približki mu*x*(1-x) za poljuben mu in N

    a=0
    b=1
    h=(b-a)*1.0/N
    
    U=[]
    
    for j in range(0,N+1):

         U.append( mu*j*h*(1-j*h) )
      
    return U

    
# ******************* newton *******************


def F(pribl,N,delta):
    
    a=0
    b=1
    h=(b-a)*1.0/N
    F=[]
    
    for j in range(1,N,1): # iteracijo začnemo v točkah ob robnih/začetnih pogojih; (torej od druge do predzadnje)
        
        F.append(-0.5*h**2.*(-(pribl[j+1]-2.*pribl[j]+pribl[j-1])/(h**2.)+ddy(pribl[j],delta)))
        
    return F


def B(pribl_j,N,delta): # elementi na diagonali
    
    a=0
    b=1
    h=(b-a)*1.0/N
    
    B=1+h**2*0.5*ddy(pribl_j,delta)
    return B
    
def J(pribl,N,delta): # Jacobijeva matrika dimenzije (N-1)x(N-1)
    
    A=-0.5 # elementi ob diagonali
    C=-0.5 # elementi ob diagonali
    
#    J=np.matrix((N-1, N-1))
    J=np.zeros((N-1, N-1))

    
    for k in range(N-1):
        for j in range(N-1):
            if j==k:
                J[j,k]=B(pribl[j+1],N,delta)  # iteracijo začnemo v točkah ob robnih/začetnih pogojih; 
            elif j+1==k:
                J[j,k]=C
    for j in range(N-1):
        for k in range(N-1):
            if k+1==j:
                J[j,k]=A
    return J



def NEWTON(pribl,N,eps,delta):
    
    Y=np.array(pribl)  # ima N+1 točk
    print(len(Y))
    dY=[eps+1 for j in range(N-1)]  # popravek približka ima eno točko manj (N-1 točk), ker ne računamo v robnih pogojih /beri dalje
    print(len(dY))

    st=0    
    while abs(max(dY))>eps:   # pogoj za iteriranje

        MJ=J(Y,N,delta)
        print(MJ.shape)
        VF=F(Y,N,delta)
        print(len(VF))
        dY=np.linalg.solve(MJ,VF)
        st=st+1
        
        for j in range(N-1):
            
            Y[j+1]=Y[j+1]+dY[j] # iteracijo računamo DO!!! predzadnje (robne) točke s tem da prve robne točke NE!! prepišemo
            
            if abs(Y[j+1])>1000:
                print("divergira")
                Y=np.zeros(N+1)
                return Y
    print(st)
    return Y , st , dY



#************************** GRAFIRANJE ****************

##_________ rešitve  TRANSCEDENTNA
#
#x=np.linspace(0,3.5,100)
#
#FIG1= plt.figure()
#ax=plt.subplot(1, 1, 1 )
#ax.plot(x,x*np.sqrt(8/0.5), 'b',label='$\delta=0.5$')#+'; $err_{max}=$'+str(errGB))
#ax.plot(x,x*np.sqrt(8/1.), 'b--',label='$\delta=1$')#+'; $err_{max}=$'+str(errGB))
#ax.plot(x,x*np.sqrt(8/3.), 'b-.',label='$\delta=3$')#+'; $err_{max}=$'+str(errGB))
#ax.plot(x,x*np.sqrt(8/3.5), 'b:',label='$\delta=3.5$')#+'; $err_{max}=$'+str(errGB))
#ax.plot(x,np.cosh(x), 'k',label='$\cos(\zeta)$')#+'; $err_{max}=$'+str(errGB))
#ax.set_xlabel( '$\zeta$' )       
##ax.set_ylabel( '$y$' )
#ax.set_title( 'Grafično reševanje transcedentne enačbe')
#ax.legend(loc='best')

##_________ Analitični rešitvi 
##
#
#D=np.array([0.5,1.,3.,3.5]) # with 4 line syles '-','--','.-',':', dashes=[12,6,12,6,3,6]
#
#X = np.linspace(a,b,N+1)   # INTERVAL enakomerni korak      
#
#FIGresAN= plt.figure()
#ANres=plt.subplot(1, 1, 1 )
#
#for j in range(0,len(D)):
#    
#    if j==0:
#        ksi1=0.2583923655030188
#        ksi2=3.2595598244395485
#        An=Analiticna(N,ksi1)
#        An2=Analiticna(N,ksi2)
#        ANres.plot(X, An, 'r', label='$\zeta_{1}$ za $\delta=$'+str(D[j]))
#        ANres.plot(X, An2, 'b',  label='$\zeta_{2}$ za $\delta=$'+str(D[j]))
#    elif j==1:
#        ksi1=0.379291149762689
#        ksi2=2.734675693030527
#        An=Analiticna(N,ksi1)
#        An2=Analiticna(N,ksi2)
#        ANres.plot(X, An, 'r--', label='$\zeta_{1}$ za $\delta=$'+str(D[j]))
#        ANres.plot(X, An2, 'b--',  label='$\zeta_{2}$ za $\delta=$'+str(D[j]))  
#    elif j==2:
#        ksi1=0.8433769410714723
#        ksi2=1.6441423148135939
#        An=Analiticna(N,ksi1)
#        An2=Analiticna(N,ksi2)
#        ANres.plot(X, An, 'r-.', label='$\zeta_{1}$ za $\delta=$'+str(D[j]))
#        ANres.plot(X, An2, 'b-.',  label='$\zeta_{2}$ za $\delta=$'+str(D[j]))
#    elif j==3:
#        ksi1=1.1379634157095866
#        ksi2=1.2635856746592544
#        An=Analiticna(N,ksi1)
#        An2=Analiticna(N,ksi2)
#        ANres.plot(X, An, 'r:', label='$\zeta_{1}$ za $\delta=$'+str(D[j]))
#        ANres.plot(X, An2, 'b:',  label='$\zeta_{2}$ za $\delta=$'+str(D[j]))
#
#
#ANres.set_xlabel( '$x$' )       
#ANres.set_ylabel( '$y$' )
#ANres.set_title( 'Analitične rešitve za različne vrednosti $\delta$')
#ANres.legend(loc='best')


#****************** IZRAČUNI ***********************


#
eps=10**(-10)
#
#w=2.
mu=1 # parameter konveksnega začetnega približka mu*x*(1-x) 
#mu=16 # metoda divergira
#
#delta=0.5
#ksi1=0.2583923655030188
#ksi2=3.2595598244395485

#delta=1.
#ksi1=0.379291149762689
#ksi2=2.734675693030527

delta=3.0
ksi1=0.8433769410714723
ksi2=1.6441423148135939

#delta=3.5
#ksi1=1.1379634157095866
#ksi2=1.2635856746592544

An=Analiticna(N,ksi1)
An2=Analiticna(N,ksi2) # ni iterativne rešitve (dobim jo z newtonv ali strelsko metodo)


#####_________napaka v odvisnosti od št točk N (delitve intervala)
##
#
NN= np.linspace(10,500,19)  # with 4 line syles '-','--','.-',':', dashes=[12,6,12,6,3,6]

FIGoptN= plt.figure()
optN=plt.subplot(2, 1, 1 )

N_it=[]
napaka=[]
SUMnapaka=[]

for j in range(0,int(len(NN))):
    
    An=Analiticna(int(NN[j]),ksi1)
    U=pribl(mu,int(NN[j]))          
    GB, Niter, errGB =NEWTON(U,int(NN[j]),eps,delta)    
    
    print('N='+str(int(NN[j]))+' -> N_it='+str(Niter))
    
    N_it.append(Niter)
    napaka.append(np.amax(abs(np.array(GB)-np.array(An))))
    SUMnapaka.append(np.linalg.norm(abs(np.array(GB)-np.array(An))))

    
optN.plot(NN, N_it, 'k:')
optN.set_xlabel( '$N_{točk}$' )       
optN.set_ylabel( '$N_{it}$' )
optN.set_title( 'Potrebno število iteracij za dosego razlike $err=$'+str(eps)+' za različno št. točk $N$ pri vrednosti $\mu=$'+str(mu)+' (Newtonova)')
optN.legend(loc='best')
#optN.set_yscale('log')

optN=plt.subplot(2, 1, 2 )
optN.plot(NN,napaka, 'k:', label='največja napaka na intervalu rešitve GB' )
optN.plot(NN,SUMnapaka, 'b-.', label='vsota vseh odstopanj od rešitve GB na intervalu' )
optN.set_title( 'Dosežena natančnost pri računanju do razlike $err=$'+str(eps)+' za različno št. točk $N$ pri $\mu=$'+str(mu)+' (Newtonova)')
optN.set_xlabel( '$N_{točk}$' )       
optN.set_ylabel( '$|GB_{num}-GB_{an}|$' )
optN.legend(loc='best')
optN.set_yscale('log')



#*********************** Variacija po mu (optimalni w=-0.47) ****************** 
#
N=50

An=Analiticna(N,ksi1)
An2=Analiticna(N,ksi2)

MU=np.array([1.,1.5,5.,10.,15.5,16.]) # with 4 line syles '-','--','.-',':', dashes=[12,6,12,6,3,6]
X = np.linspace(a,b,N+1)  # INTERVAL enakomerni korak      

FIGmu= plt.figure()
zacetni=plt.subplot(1, 1, 1 )
#
for j in range(0,len(MU)):   
    
    U=pribl(MU[j],N)
    GB, Niter, errGB =NEWTON(U,N,eps,delta)

    if j==0:
        zacetni.plot(X, U, 'c--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'c:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        print('prva')    
    elif j==1:
        zacetni.plot(X, U, 'm--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'm:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        print('druga')    
    elif j==2:
        zacetni.plot(X, U, 'b--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'b:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        print('tretja')    
    elif j==3:
        zacetni.plot(X, U, 'g--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'g:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        print('cetrta')    
    elif j==4:
        zacetni.plot(X, U, 'r--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'r:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        print('peta') 
    elif j==5:
        zacetni.plot(X, U, 'y--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'y:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
        print('sesta') 
        
#zacetni.plot(X, U, 'k:', label='začetni približek $\mu=$'+str(mu))
zacetni.plot(X, An, 'k--', label='analitična rešitev za $\zeta_{1}$')
zacetni.plot(X, An2, 'k-.', label='analitična rešitev za $\zeta_{2}$')


zacetni.set_xlabel( '$x$' )
zacetni.set_ylabel( '$y$' )
zacetni.set_title( 'Diferenčne rešitve Gelfand-Bratujevega robnega problema z $\delta=$'+str(delta)+' pri različnih $\mu$ (Newtonova)')

zacetni.legend(loc='upper right')
#zacetni.legend(loc='best')
#zacetni.set_yscale('log')
#
FIGerrMU= plt.figure()
MUnapake=plt.subplot(1, 1, 1 )

for j in range(0,len(MU)):   
    
    U=pribl(MU[j],N)
    GB, Niter, errGB =NEWTON(U,N,eps,delta)
    
    if j==0:
        MUnapake.plot(X, abs(GB-An), 'c:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('prva')    
    elif j==1:
        MUnapake.plot(X, abs(GB-An), 'm:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('druga')    
    elif j==2:
        MUnapake.plot(X, abs(GB-An), 'b:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('tretja')    
    elif j==3:
        MUnapake.plot(X, abs(GB-An2), 'g:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('cetrta')    
    elif j==4:
        MUnapake.plot(X, abs(GB-An2), 'r:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('peta') 
    elif j==5:
        MUnapake.plot(X, abs(GB-An2), 'y:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('sesta') 
        
MUnapake.set_xlabel( '$x$' )       
MUnapake.set_ylabel( '$|GB_{num}-GB_{an}|$' )
MUnapake.set_title( 'Natančnost diferenčne metode z $\delta=$'+str(delta)+' pri različnih $\mu$ in pogoju ob $err=$'+str(eps)+' (Newtonova)')
MUnapake.legend(loc='upper right')
#MUnapake.legend(loc='best')
MUnapake.set_yscale('log')