# -*- coding: utf-8 -*-

import scipy 
import numpy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import special
import pylab
import time

#pi=3.1415926535897932384626433
#i=complex(0,1)
#a=0.01
#B=0.0000993333333333333      #na Beta iz enaèbe
#************************************
#f=open("tabela_lastnih_1.txt", "w")
#f.write("%5f %10f %10f %10f %10f %10f" %(i, Neki[0], Neki[1], Neki[2], Neki[3], Neki[4]))
#f.write("\n")
#f.close()
#****************************************
#import time
#tic = time.clock()
#toc = time.clock()
#toc-tic
#****************************************
#def BRN(N):
#    d=3.0/N
#    E=[(a+j*d)/(scipy.sqrt(B+2.0/3*(a+j*d)**3)) for j in range(N)]
#    return E
#
#def f(x,y):#(odvod Bernullijeve enaèbe)
#    M=10**6
#    if x>M:
#        return 1
#    if y>M:
#        return 1
#    return -y**3+y*1.0/(a+x)


def f( x, y ):
    xzun=-5.
    k=0.1
    A=1.
    pi=3.1415926535897932384626433
    return -k*(y-xzun)+A*numpy.sin(2.*pi*(x-10.)/24.)
    
    
#def RK4(N):
#    d=3.0/N
#    E=[0 for j in range(N)]
#    E[0]=1
#    for j in range(0,N-1,1):
#        if abs(E[j])>10**8:
#            E[j]=0
#            return E
#        k1=d*f(j*d,E[j])
#        k2=d*f(j*d+0.5*d, E[j]+0.5*k1)
#        k3=d*f(j*d+0.5*d, E[j]+0.5*k2)
#        k4=d*f((j+1)*d,E[j]+k3)
#        E[j+1]=E[j]+1.0/6*(k1+2*k2+2*k3+k4)
#    return E
#*****************************************
#*****************************************
def AD4():          #funkciji podamo maksimalno napako
    lim=10000000
    err1=10**(-8)           #spodnja napaka
    err2=10**(-15)          #zgornja napaka
    X=[0 for j in range(lim)]  #tabela x-ov
    E=[0 for j in range(lim)]  #tabela vrednosti, na istem mestu kot lastni x
    E[0]=21                     #zaèetni pogoj
    H=1000
    n=0
    while ((X[n])<3.0 and n<(lim-1)):  #tole vrtimo dokler ne pridemo do x=3
        k1=H*f(X[n],E[n])
        k2=H*f(X[n]+0.5*H, E[n]+0.5*k1)
        k3=H*f(X[n]+0.5*H, E[n]+0.5*k2)
        k4=H*f(X[n]+H,E[n]+k3)
        r1=E[n]+1.0/6*(k1+2*k2+2*k3+k4)
        #*****
        k1=H*0.5*f(X[n],E[n])
        k2=H*0.5*f(X[n]+0.25*H, E[n]+0.5*k1)
        k3=H*0.5*f(X[n]+0.25*H, E[n]+0.5*k2)
        k4=H*0.5*f(X[n]+0.5*H,E[n]+k3)
        r2=E[n]+1.0/6*(k1+2*k2+2*k3+k4)
        #****
        x1=X[n]+H*0.5
        k1=H*0.5*f(x1,r2)
        k2=H*0.5*f(x1+0.25*H, r2+0.5*k1)
        k3=H*0.5*f(x1+0.25*H, r2+0.5*k2)
        k4=H*0.5*f(x1+H*0.5,r2+k3)
        r3=r2+1.0/6*(k1+2*k2+2*k3+k4)
        if abs(r3-r1)<err1:
            X[n+1]=X[n]+H
            E[n+1]=r3
            if abs(r3-r1)<err2:
                H=2*H               #veèam korak, èe je natanènost prevelika
            n=n+1
        else:           #manjam korak, èe je natanènost premajhna
            H=0.5*H
    if n>(lim-2):
        print("Prekratka tabela!")
    return X,E

#*****************************************
#*****************************************


"""
j=2
z=0
A1=[0 for k in range(110)] #tabela natanènosti
A2=[0 for k in range(110)] #tabela èasovne zahtevnosti od j
while j<100:
    tic = time.clock()
    X,E=AD4(10**(-j*0.1))         #Naredimo proces v odvisnoti od natanènosti   
    tac = time.clock()
    A2[z]=scipy.log10(tac-tic)
    ab=len(E)
    print(ab)
    B=BRN(ab)
    T=[0 for p in range(ab)]    #tabela napake
    for y in range(ab):
        T[y]=abs(E[y]-B[y])
    A1[z]=scipy.log10(max(T))
    print(j)
    j=j+3
    z=z+1

A3=[0 for k in range(105)]
A4=[0 for k in range(105)]
A3[0]=A1[0]
A4[0]=A2[0]

for k in range(1,105,1):
    if A1[k]==0:
        A3[k]=A3[k-1]
    else:
        A3[k]=A1[k]
    
for k in range(1,105,1):
    if A2[k]==0:
        A4[k]=A4[k-1]
    else:
        A4[k]=A2[k]
"""
#*****************************************
#**************************************
#B=BRN(5000)
x=[k/100 for k in range(10000)]
X,E=AD4()

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(A3,A4,"b")
ax.plot(X,E,"k-")
#leg = ax.legend(('Analiticna resitev','E1(N=80)','E1(N=160)','E3(N=80)','E3(N=160)','E4(N=80)', 'E4(N=160)', 'E5(N=80)', 'E5(N=160)', 'RK4(N=80)', 'RK4(N=160)'), 'best')
plt.xlabel('Meja za napako: 10^(x-os)')
plt.ylabel('log10(cas)')
#plt.ylabel('Log10(relativna napaka)')
plt.show()

