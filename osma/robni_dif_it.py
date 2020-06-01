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

delta=1.
ksi1=0.379291149762689
ksi2=2.734675693030527

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

def ITER(A,N,eps,w,delta):  

    a=0
    b=1
    n=len(A)
    
    h=(b-a)*1.0/N
    
    d=eps+1
    Y=np.array(A)
#    B=np.array([0 for j in range(n)])
    st=0 
    
    d=eps+1  # pogoj za vstop v zanko
    while d>eps: # pogoj za iteriranje
    
        st=st+1       
#        print('moj:'+str(d))
#        d=0
#        for j in range(n):
#            d=abs(Y[j]-B[j])+d
#        print(d)
        B=np.array(Y)
        
        for j in range(1,n-1,1):   # iteracijo začnemo v točkah ob robnih/začetnih pogojih; (torej od druge do predzadnje)
        
            Y[j]=1/(1+w)*(0.5*(Y[j+1]+Y[j-1])+w*Y[j]-h**2*0.5*ddy(Y[j],delta))

            if Y[j]>1000:                
                Y=[-1 for k in range(n)]
                print("divergira")                
                return Y,st,d
                
        d=np.amax(abs(Y-B)) # največje odstopanje med zadnjima iterativnima korakoma na intervalu rešitve

#        if st>3:
#            print(B)
#            print(d)
#            print(Y)
#            break
                
    return Y,st,d
    
    

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

######## ******************  Analitični rešitvi KONVERGENČNI PARAMETER ****************** 
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


###### ****************** IZRAČUNI ***** Variaciija po w ******************


#
eps=10**(-10)
#
#w=2.
mu=1 # parameter konveksnega začetnega približka mu*x*(1-x) 
#mu=16 # metoda divergira

#delta=0.5
#ksi1=0.2583923655030188
#ksi2=3.2595598244395485

delta=1.
ksi1=0.379291149762689
ksi2=2.734675693030527

#delta=3.0
#ksi1=0.8433769410714723
#ksi2=1.6441423148135939

#delta=3.5
#ksi1=1.1379634157095866
#ksi2=1.2635856746592544

An=Analiticna(N,ksi1)
#An=Analiticna(N,ksi2) # ni iterativne rešitve (dobim jo z newtonv ali strelsko metodo)


##### ******************  optimalni KONVERGENČNI PARAMETER w  (pri N=50 /sicer neodvisen od N) ****************** 

#
#T = np.linspace(-2.,0.5,101) 
#
#FIGoptW= plt.figure()
#optW=plt.subplot(2, 1, 1 )
#
#N_it=[]
#napaka=[]
#SUMnapaka=[]
#
#for j in range(0,len(T)):
#    U=pribl(mu,N)      
#    GB, Niter, errGB =ITER(U,N,eps,T[j],delta)    
#    print('w='+str(T[j])+' -> N_it='+str(Niter))
#    
#    N_it.append(Niter)
#    napaka.append(np.amax(abs(np.array(GB)-np.array(An))))
#    SUMnapaka.append(np.linalg.norm(abs(np.array(GB)-np.array(An))))
#
#    
#optW.plot(T, N_it, 'k:')
#optW.set_xlabel( '$\omega$' )       
#optW.set_ylabel( '$N_{it}$' )
#optW.set_title( 'Potrebno število iteracij za dosego razlike $err=$'+str(eps)+' pri vrednosti $\mu=$'+str(mu)+' (iterativno)')
#optW.legend(loc='best')
#optW.set_yscale('log')
#
#devW=plt.subplot(2, 1, 2 )
#devW.plot(T,napaka, 'k:', label='največja napaka na intervalu rešitve GB' )
#devW.plot(T,SUMnapaka, 'b:', label='vsota vseh odstopanj od rešitve GB na intervalu' )
#devW.set_title( 'Odstopanja od analitične rešitve pri računanju do razlike $err=$'+str(eps)+' za $\mu=$'+str(mu)+' (iterativno)')
#devW.set_xlabel( '$\omega$' )       
#devW.set_ylabel( '$|GB_{num}-GB_{an}|$' )
#devW.legend(loc='best')
#
#devW.set_yscale('log')


##### ******************  napaka v odvisnosti od št točk delitve intervala N  (pri w=-.47 ) ****************** 

#
#NN= np.linspace(10,100,30)  # with 4 line syles '-','--','.-',':', dashes=[12,6,12,6,3,6]
#
#FIGoptN= plt.figure()
#optN=plt.subplot(2, 1, 1 )
#
#N_it=[]
#napaka=[]
#SUMnapaka=[]
#
#for j in range(0,int(len(NN))):
#    
#    An=Analiticna(int(NN[j]),ksi1)
#    U=pribl(mu,int(NN[j]))          
#    GB, Niter, errGB =ITER(U,int(NN[j]),eps,-0.47,delta)    
#    
#    print('N='+str(int(NN[j]))+' -> N_it='+str(Niter))
#    
#    N_it.append(Niter)
#    napaka.append(np.amax(abs(np.array(GB)-np.array(An))))
#    SUMnapaka.append(np.linalg.norm(abs(np.array(GB)-np.array(An))))
#
#    
#optN.plot(NN, N_it, 'k:')
#optN.set_xlabel( '$N_{točk}$' )       
#optN.set_ylabel( '$N_{it}$' )
#optN.set_title( 'Potrebno število iteracij za dosego razlike $err=$'+str(eps)+' za različno št. točk $N$ pri vrednosti $\mu=$'+str(mu)+' (iterativno)')
#optN.legend(loc='best')
#optN.set_yscale('log')
#
#optN=plt.subplot(2, 1, 2 )
#optN.plot(NN,napaka, 'k:', label='največja napaka na intervalu rešitve GB' )
#optN.plot(NN,SUMnapaka, 'b:', label='vsota vseh odstopanj od rešitve GB na intervalu' )
#optN.set_title( 'Dosežena natančnost pri računanju do razlike $err=$'+str(eps)+' za različno št. točk $N$ pri $\mu=$'+str(mu)+' (iterativno)')
#optN.set_xlabel( '$N_{točk}$' )       
#optN.set_ylabel( '$|GB_{num}-GB_{an}|$' )
#optN.legend(loc='best')
#
#optN.set_yscale('log')


#####_____________ REŠITVE variacije po w  (pri N=50 /sicer neodvisen od N)  ****************** 

#W=np.array([-0.47,0.,0.01,0.05,0.1,1.]) # with 4 line syles '-','--','.-',':', dashes=[12,6,12,6,3,6]
#X = np.linspace(a,b,N+1)  # INTERVAL enakomerni korak      
#
#FIGw= plt.figure()
#resitve=plt.subplot(1, 1, 1 )
##
#for j in range(0,len(W)):   
#    
#    U=pribl(mu,N)
#    GB, Niter, errGB =ITER(U,N,eps,W[j],delta)
#
#    if j==0:
#        resitve.plot(X, GB, 'c:', label='$\omega=$'+str(W[j])+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
#        print('prva')    
#    elif j==1:
#        resitve.plot(X, GB, 'm:', label='$\omega=$'+str(W[j])+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
#        print('druga')    
#    elif j==2:
#        resitve.plot(X, GB, 'b:', label='$\omega=$'+str(W[j])+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
#        print('tretja')    
#    elif j==3:
#        resitve.plot(X, GB, 'g:',  label='$\omega=$'+str(W[j])+'; $N_{it}=$'+str(Niter))#)+'; $err_{max}=$'+str(errGB))
#        print('cetrta')    
#    elif j==4:
#        resitve.plot(X, GB, 'r:', label='$\omega=$'+str(W[j])+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
#        print('peta') 
#    elif j==5:
#        resitve.plot(X, GB, 'y:', label='$\omega=$'+str(W[j])+'; $N_{it}=$'+str(Niter))#+'; $err_{max}=$'+str(errGB))
#        print('sesta') 
#        
#resitve.plot(X, U, 'k:', label='začetni približek $\mu=$'+str(mu))
#resitve.plot(X, An, 'k--', label='analitična rešitev')
#
#resitve.set_xlabel( '$x$' )
#resitve.set_ylabel( '$y$' )
#resitve.set_title( 'Diferenčne rešitve Gelfand-Bratujevega robnega problema za $\delta=$'+str(delta)+' pri različnih $\omega$ in pogoju  $err=$'+str(eps)+' (iterativno)')
##resitve.legend(loc='upper right')
#resitve.legend(loc='best')
##resitve.set_yscale('log')
##
####_____________ NAPAKE variacije po w  (pri N=50 /sicer neodvisen od N)
##
#FIGerr= plt.figure()
#napake=plt.subplot(1, 1, 1 )
##
#for j in range(0,len(W)):   
#    
#    if j==0:
#        napake.plot(X, abs(GB-An), 'c:', label='$\omega=$'+str(W[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('prva')    
#    elif j==1:
#        napake.plot(X, abs(GB-An), 'm:', label='$\omega=$'+str(W[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('druga')    
#    elif j==2:
#        napake.plot(X, abs(GB-An), 'b:', label='$\omega=$'+str(W[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('tretja')    
#    elif j==3:
#        napake.plot(X, abs(GB-An), 'g:', label='$\omega=$'+str(W[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('cetrta')    
#    elif j==4:
#        napake.plot(X, abs(GB-An), 'r:', label='$\omega=$'+str(W[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('peta') 
#    elif j==5:
#        napake.plot(X, abs(GB-An), 'y:', label='$\omega=$'+str(W[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('sesta') 
#        
#napake.set_xlabel( '$x$' )       
#napake.set_ylabel( '$|GB_{num}-GB_{an}|$' )
#napake.set_title( 'Natančnost diferenčne metode za $\mu=$'+str(mu)+' pri različnih $\omega$ in pogoju  $err=$'+str(eps)+' (iterativno)')
##resitve.legend(loc='upper right')
#napake.legend(loc='best')
#napake.set_yscale('log')


######*********************** REŠITVE Variacija po mu (optimalni w=-0.47, N=50) ****************** 

MU=np.array([1.,1.5,5.,10.,15.5,16.]) # with 4 line syles '-','--','.-',':', dashes=[12,6,12,6,3,6]
X = np.linspace(a,b,N+1)  # INTERVAL enakomerni korak      

FIGmu= plt.figure()
zacetni=plt.subplot(1, 1, 1 )
#
for j in range(0,len(MU)):   
    
    U=pribl(MU[j],N)
    GB, Niter, errGB =ITER(U,N,eps,-0.47,delta)

    if j==0:
        zacetni.plot(X, U, 'c--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'c:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('prva')    
    elif j==1:
        zacetni.plot(X, U, 'm--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'm:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('druga')    
    elif j==2:
        zacetni.plot(X, U, 'b--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'b:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('tretja')    
    elif j==3:
        zacetni.plot(X, U, 'g--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'g:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('cetrta')    
    elif j==4:
        zacetni.plot(X, U, 'r--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'r:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('peta') 
    elif j==5:
        zacetni.plot(X, U, 'y--', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        zacetni.plot(X, GB, 'y:', label='$\mu=$'+str(MU[j])+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('sesta') 
        
#zacetni.plot(X, U, 'k:', label='začetni približek $\mu=$'+str(mu))
zacetni.plot(X, An, 'k--', label='analitična rešitev')

zacetni.set_xlabel( '$x$' )
zacetni.set_ylabel( '$y$' )
zacetni.set_title( 'Diferenčne rešitve Gelfand-Bratujevega robnega problema pri različnih $\mu$ (iterativno)')

#zacetni.legend(loc='upper right')
zacetni.legend(loc='best')
#zacetni.set_yscale('log')

###_____________ NAPAKE variacije po mu (optimalni w=-0.47, N=50) 

FIGerrMU= plt.figure()
MUnapake=plt.subplot(1, 1, 1 )

for j in range(0,len(MU)):   
    
    U=pribl(MU[j],N)
    GB, Niter, errGB =ITER(U,N,eps,-0.47,delta)
    
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
        MUnapake.plot(X, abs(GB-An), 'g:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('cetrta')    
    elif j==4:
        MUnapake.plot(X, abs(GB-An), 'r:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
        print('peta') 
#    elif j==5:
#        MUnapake.plot(X, abs(GB-An), 'y:', label='$\mu=$'+str(MU[j]))#+'; $N_{it}=$'+str(Niter)+'; $err_{max}=$'+str(errGB))
#        print('sesta') 
        
MUnapake.set_xlabel( '$x$' )       
MUnapake.set_ylabel( '$|GB_{num}-GB_{an}|$' )
MUnapake.set_title( 'Natančnost diferenčne metode pri različnih $\mu$ in pogoju ob $err=$'+str(eps)+' (iterativno)')
#resitve.legend(loc='upper right')
MUnapake.legend(loc='best')
MUnapake.set_yscale('log')
