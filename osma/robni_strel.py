# -*- coding: utf-8 -*-
import scipy as sc
import numpy  as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#*************************************************
pi=3.1415926535897932384626433

N=50
a=0
b=1
h=(b-a)*1.0/N  # koraki med točkami

eps=10**(-10)


def Analiticna(t,ksi): #analitična rešitev enačbe G-B
    
    A=[]
    for j in range(len(t)):
        
        A.append(-2*sc.log( sc.cosh( ksi*(1-2.*t[j]))/sc.cosh(ksi) ) )
        
    return A

    
#************************** bisekcija ****************
    
def difuzija(y, t):
    
#    delta=0.5
#    delta=1.
    delta=3.5

    xt, dxt = y

    ddxt = [dxt, -delta* sc.exp(xt) ]
    return ddxt


def BISEK1(ddxt, t, xt0, dxt0, eps):
    
    st=0 
    d=eps+1   #pogoj za vstop v zanko
    priblizki=[]
    
    zg=dxt0+dxt0/2
    sp=dxt0-dxt0/2
    
    solzg = odeint(difuzija, [0, zg], t) # testni strel 
    solsp = odeint(difuzija, [0, sp], t) # testni strel 
    Xtzg = solzg[:,0]
    Xtsp = solsp[:,0]

    Nzg=len(Xtzg)-1 
    Nsp=len(Xtsp)-1
    if Xtzg[Nzg]>0 and Xtsp[Nsp]>0:
        print('izberi manjši približek začetnega odvoda')
    elif Xtzg[Nzg]<0 and Xtsp[Nsp]<0:
        print('izberi večji približek začetnega odvoda')
    
    else:
        while d>eps:
            print(dxt0)

            priblizki.append(dxt0)    
            sol = odeint(difuzija, [0, dxt0], t) # testni strel 
            Xt = sol[:,0]
            dXt = sol[:,1]
#            Xt, dXt, t ,H =control(ddxt, delta, eps, xt0, dxt0, t0, tmax ) # strel za iteracijo
            
            N=len(Xt)-1
            if Xt[N]>0:
                zg=(zg+sp)/2.
                
            elif Xt[N]<0:
                sp=(zg+sp)/2.
                
            dxt0=(zg+sp)/2 
            
            st=st+1    #štejem število trelov
            d=abs(Xt[N]) #napaka vrednosti xt v zadnji/robni točki
#            print(d)
            
        return Xt,dXt,t,st,priblizki,d



def BISEK2(ddxt, t, xt0, dxt0, eps):
    
    st=0 
    d=eps+1   #pogoj za vstop v zanko
    priblizki=[]
    
    zg=dxt0+dxt0/2
    sp=dxt0-dxt0/2
    
    solzg = odeint(difuzija, [0, zg], t) # testni strel 
    solsp = odeint(difuzija, [0, sp], t) # testni strel 
    Xtzg = solzg[:,0]
    Xtsp = solsp[:,0]

    Nzg=len(Xtzg)-1 
    Nsp=len(Xtsp)-1
    if Xtzg[Nzg]>0 and Xtsp[Nsp]>0:
        print('izberi manjši približek začetnega odvoda')
    elif Xtzg[Nzg]<0 and Xtsp[Nsp]<0:
        print('izberi večji približek začetnega odvoda')
    
    else:
        while d>eps:
            print(dxt0)

            priblizki.append(dxt0)    
            sol = odeint(difuzija, [0, dxt0], t) # testni strel 
            Xt = sol[:,0]
            dXt = sol[:,1]
#            Xt, dXt, t ,H =control(ddxt, delta, eps, xt0, dxt0, t0, tmax ) # strel za iteracijo
            
            N=len(Xt)-1
            if Xt[N]<0:
                zg=(zg+sp)/2.
                
            elif Xt[N]>0:
                sp=(zg+sp)/2.
                
            dxt0=(zg+sp)/2 
            
            st=st+1    #štejem število trelov
            d=abs(Xt[N]) #napaka vrednosti xt v zadnji/robni točki
#            print(d)
            
        return Xt,dXt,t,st,priblizki,d


t = np.linspace(a,b,N+1) 

dxt01=2.5
GB1, odvod1, koraki1, Nit1, priblizki1, err=BISEK1(difuzija, t, a, dxt01, eps)

dxt02=8
GB2, odvod2, koraki2, Nit2, priblizki2, err=BISEK2(difuzija, t, a, dxt02, eps)

#GB, koraki, Nit, err=BISEK(ddxt, eps, a, dxt0, a, b)

#delta=0.5
#ksi1=0.2583923655030188
#ksi2=3.2595598244395485
#
#delta=1.
#ksi1=0.379291149762689
#ksi2=2.734675693030527

#delta=3.0
#ksi1=0.8433769410714723
#ksi2=1.6441423148135939

delta=3.5
ksi1=1.1379634157095866
ksi2=1.2635856746592544  
    
An=Analiticna(koraki1,ksi1)
An2=Analiticna(koraki2,ksi2)


# grafiranje analitična in strel

FIGAn= plt.figure()
Anfig1=plt.subplot(3, 1, 1 )
plt.plot(koraki1, An, 'r--', label='analitična za $\zeta_{1}$')
plt.plot(koraki2, An2, 'b--', label='analitična za $\zeta_{2}$')
plt.plot(koraki1, GB1, 'r:', label='strelska $\zeta_{1}$')
plt.plot(koraki2, GB2, 'b:', label='strelska $\zeta_{2}$')
plt.legend(loc='upper right')
plt.ylabel('y')
plt.xlabel('x')
plt.title('Rešitev Gelfand-Bratujevega robnega problema za $\delta=$'+str(delta)+' pri pogoju $err=$'+str(eps)+' (strelska)')
plt.show()

Anfig2=plt.subplot(3, 1, 2 )
plt.plot(koraki1, abs(GB1-An), 'r:', label='strelska $\zeta_{1}$')
plt.plot(koraki2, abs(GB2-An2), 'b:', label='strelska $\zeta_{2}$')
plt.legend(loc='upper right')
plt.ylabel('y')
plt.xlabel('x')
Anfig2.set_yscale('log')
plt.title('Odstopanja od analitične rešitve za $\delta=$'+str(delta)+' pri pogoju $err=$'+str(eps)+' (strelska)')
plt.show()

pribfig=plt.subplot(3, 1, 3 )
plt.plot(priblizki1, 'r--', label='strelska $y\'_{0}=$'+str(dxt01)+'; $N_{it}=$'+str(Nit1))
plt.plot(priblizki2, 'b-.', label='strelska $y\'_{0}=$'+str(dxt02)+'; $N_{it}=$'+str(Nit2))
plt.title('Približki odvoda v prvi točki strela tekom iteracije pri pogoju bisekcije $err=$'+str(eps))
plt.legend(loc='right')
plt.ylabel('$y\'_{0}$')
plt.xlabel('korak iteracije bisekcije')
plt.show()

