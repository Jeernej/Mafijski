# -*- coding: utf-8 -*-

import scipy as sc
import numpy as np
from math import *
from sys import *
from numpy import random      # random.uniform(low=0.0, high=1.0, size=None) 
from numpy import mean
from numpy import std
from numpy import median

#def mad(a):
#    c=Gaussian.ppf(3/4.)
#    axis=0
#    center=np.median
#    # c \approx .6745
#    """
#    The Median Absolute Deviation along given axis of an array
#
#    Parameters
#    ----------
#    a : array-like
#        Input array.
#    c : float, optional
#        The normalization constant.  Defined as scipy.stats.norm.ppf(3/4.),
#        which is approximately .6745.
#    axis : int, optional
#        The defaul is 0. Can also be None.
#    center : callable or float
#        If a callable is provided, such as the default `np.median` then it
#        is expected to be called center(a). The axis argument will be applied
#        via np.apply_over_axes. Otherwise, provide a float.
#
#    Returns
#    -------
#    mad : float
#        `mad` = median(abs(`a` - center))/`c`
#    """
#    a = np.asarray(a)
#    if callable(center):
#        center = np.apply_over_axes(center, a, axis)
#    return np.median((np.fabs(a-center))/c, axis=axis)
    
def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
#    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))
    
#___________________________________________________________________________

pi=3.1415926535897932384626433

# sprehodi v sfericnem koordinatnem sistemu za različne mi
def sprehodSFER(mu, koraki):
     
    x,y = 0,0 
    X,Y = [x],[y]
  
    
    for j in range(0, koraki):
        
        l = 1 / (random.uniform() ** (1./ (mu-1.)))  # l**(-mi) poenostavimo v Paretovo porazdelitev
        phi = random.uniform(0, 2 * pi)
        
        x=x + l * sc.cos(phi)
        y=y + l * sc.sin(phi)
        
        X.append(x)
        Y.append(y)
    
    #R=np.sqrt(np.array(X)**2+np.array(Y)**2)

    return X,Y#,R
    
def koncneTockeSFER(mi, koraki, sprehodi):
         
    X,Y,R = [],[],[]
#    sigX,sigY,sigR = [],[],[]

    while len(X) < sprehodi:  # opravim veliko sprehodov za dobro statistiko
            
        #x,y,r=sprehodSFER(mi, koraki)
        x,y=sprehodSFER(mi, koraki)
        
        if abs(x[koraki])<200001. and abs(y[koraki])<200001.: # vpišem le ne preoddaljene končne točke sprehodov 
            
            X.append(x[koraki])  # (drugače imam memory problem pri histogramih)
            Y.append(y[koraki])
#            R.append(r[koraki])
            
#        sigX.append(np.std(X))  # vpišem končne točke sprehodov
#        sigY.append(np.std(Y))
#        sigR.append(np.std(R))
        
    sigX=np.std(X,dtype=np.float64)  # deviacija končih točk sprehodov
    sigY=np.std(Y,dtype=np.float64)
#    sigR=np.std(R,dtype=np.float64)
       
    madnX=1.4826*mad(X)  # sigma=MADN=1.4826*MAD robustna deviacija končih točk sprehodov
    madnY=1.4826*mad(Y)  
#    madnR=1.4826*mad(R)     
       
    return X,sigX,madnX,Y,sigY,madnY#,R,sigR,madnR


    
    
# sprehodi v kartezicnem koordinatnem sistemu za različne f(x)
def f(x):
    Funkcije = {0: x, 1: abs(x), 2: 1./x, 3: 1./abs(x), 4: np.log(1+x)*x**4}
    return Funkcije
    
def sprehodKART(funkcija, koraki):
     
    x,y = 0,0 
    X,Y = [x],[y]
  
    
    for j in range(0, koraki):
        
        x=x+ f(random.uniform(-0.5,0.5))[funkcija]
        y=y+ f(random.uniform(-0.5,0.5))[funkcija]
        
        X.append(x)
        Y.append(y)
    
#    R=np.sqrt(np.array(X)**2+np.array(Y)**2)

    return X,Y#,R

    
def koncneTockeKART(funkcija, koraki, sprehodi):

    X,Y,R = [],[],[]
#    sigX,sigY,sigR = [],[],[]
           
    while len(X) < sprehodi:  # opravim veliko sprehodov za dobro statistiko
         
        #x,y,r=sprehodKART(funkcija, koraki)
        x,y=sprehodKART(funkcija, koraki)
        
        if abs(x[koraki])<2000001. and abs(y[koraki])<2000001.: # vpišem le ne preoddaljene končne točke sprehodov 
  
            X.append(x[koraki])  # vpišem končne točke sprehodov
            Y.append(y[koraki])
    #        R.append(r[koraki])
#
#        sigX.append(np.std(X))  # vpišem končne točke sprehodov
#        sigY.append(np.std(Y))
#        sigR.append(np.std(R))
        
    sigX=np.std(X,dtype=np.float64)  # deviacija končih točk sprehodov
    sigY=np.std(Y,dtype=np.float64)
#    sigR=np.std(R,dtype=np.float64)

    madnX=1.4826*mad(X)  # sigma=MADN=1.4826*MAD robustna deviacija končih točk sprehodov
    madnY=1.4826*mad(Y)  
#    madnR=1.4826*mad(R)       
       
    return X,sigX,madnX,Y,sigY,madnY#,R,sigR,madnR
    

    
#__________________ IZRACUNI  in  GRAFIRANJE ______________________________________    

import matplotlib.pyplot as plt

def StStolp(X): # pravilo Freedmana in Diaconisa za širino stolpcev v histogramu (iz Rač. Orod. v fiziki)

    nx=len(X)
    q1=np.percentile(X, 25) ## vrednost, pri kateri je 25% podatkov manjših od vrednosti
    q3=np.percentile(X, 75)
    iqr = q3 - q1
    sirina = 2 * iqr * nx ** (-1. / 3.)
    n=int((max(X) - min(X)) / sirina)
         
    return n
    
#________________________________________________________    
  
    
### izris sprehodov v sferičnih koordinatah 

#for k in range (0,3):
#    mu=1.5+k*1
#    
#    FIG= plt.figure()    
#    for l in range(1,5):                    
#        koraki = 10**l
#        
#        #X,Y,R=sprehodSFER(mu, koraki)
#        X,Y=sprehodSFER(mu, koraki)
#        
#        plt.subplot(2, 2, l )
#        plt.plot(X, Y, 'k')
#        plt.ylabel('y')
#        plt.xlabel('x')
#        plt.title('Sprehod pri '+str(koraki)+' korakih ($\mu=$'+str(mu)+')')
    
#
#
### izris sprehodov v kartezičnih koordinatah 
for k in range (0,5):
    funkcija=k
    FUNKCIA=['$x$','$|x|$','$1/x$','$1/|x|$','$x^{4}\ln|1+x|$']
    FIG= plt.figure()    
    for l in range(1,5):                    
        koraki = 10**l
        
        X,Y=sprehodKART(funkcija, koraki)

        plt.subplot(2, 2, l )
        plt.plot(X, Y, 'k')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.title('Sprehod pri '+str(koraki)+' korakih za '+FUNKCIA[k])
              

##### izris histogramov porazdelitve končnih točk  v sferičnih koordinatah 
#sprehodi=10**4 # število sprehodov za dobro statistiko
#
#MU=[]
#GAMAsigx=[]
#GAMAmadnx=[]
#GAMAsigy=[]
#GAMAmadny=[]
#
#for k in range (0,9): 
#    mu=1.5+k*0.25 
#    MU.append(mu)
#    print(mu)      
#    
#    #X,sigx,madnx,Y,sigy,madny,R,sigr,madnr=koncneTockeSFER(mu, 1, sprehodi)
#    #sigX,sigY,sigR = [sigx],[sigy],[sigr]
#    #madnX,madnY,madnR = [madnx],[madny],[madnr]
#    X,sigx,madnx,Y,sigy,madny=koncneTockeSFER(mu, 1, sprehodi)
#    sigX,sigY = [sigx],[sigy]
#    madnX,madnY = [madnx],[madny]
#    
##    if mu==3.5 or mu==2.5 or mu==1.5: # izris histogramov porazdelitve končnih točk sprehodov in grafov razmazanosti //za izbrane mu
##        
##        
##        KORAKI=[1,5,10,50,100,500,1000]
##        for l in range (1,7):#7
##            koraki=KORAKI[l]
##            
##            #X,sigx,madnx,Y,sigy,madny,R,sigr,madnr=koncneTockeSFER(mu, koraki, sprehodi)
##            X,sigx,madnx,Y,sigy,madny=koncneTockeSFER(mu, koraki, sprehodi)
##            
##            sigX.append(sigx)  # vpišem deviacije sprehodov
##            sigY.append(sigy)
###            sigR.append(sigr)    
##            madnX.append(madnx)  # vpišem robustne deviacije sprehodov
##            madnY.append(madny)
###            madnR.append(madnr)     
##            
##            if koraki==100:
##                X10=X
##                Y10=Y
##                npx10=StStolp(X)
##                npy10=StStolp(Y)
##                
##            elif koraki==500:
##                X100=X
##                Y100=Y
##                npx100=StStolp(X)
##                npy100=StStolp(Y)
##        
##        npx=StStolp(X)
##        npy=StStolp(Y)
##
##        FIG= plt.figure() # izris histogramov porazdelitve končnih točk za izaračun s 100,1000 in 10000 koraki     
##        plt.hist(X10, npx10, normed=True, histtype='step', color='m', label='100 korakov $(x)$') #izris histograma
##        plt.hist(Y10, npy10, normed=True, histtype='step', color='c', label='100 korakov $(y)$')#izris histograma
##        plt.hist(X100, npx100, normed=True, histtype='step', color='r', label='500 korakov $(x)$') #izris histograma
##        plt.hist(Y100, npy100, normed=True, histtype='step', color='b', label='500 korakov $(y)$')#izris histograma
##        plt.hist(X, npx, normed=True, histtype='step', color='y', label='1000 korakov $(x)$') #izris histograma
##        plt.hist(Y, npy, normed=True, histtype='step', color='g', label='1000 korakov $(y)$')#izris histograma
##        #    npr=StStolp(R)
##        #    plt.hist(R, npr, normed=False, histtype='step', color='k', label='koordinate r')#izris histograma
##        plt.title('Histogram porazdelitve končnih vrednosti koordinat pri '+str(sprehodi)+' sprehodih ($\mu=$'+str(mu)+')')
##        plt.xlabel('$x_n$')
##        plt.ylabel('N')
##        plt.legend(loc='best')
##  
##        FIG2= plt.figure() # izris grafa razmazanosti z daljšanjem časa/KORAKOI=[1,100,500,1000,2000,5000,10000]
##        plt.plot(KORAKI, np.array(sigX)**2, 'r--x', label='$\sigma^2_x(t)$')
##        plt.plot(KORAKI, np.array(madnX)**2, 'r:x', label='$MADN^2_x(t)$')
##        plt.plot(KORAKI, np.array(sigY)**2, 'b--x', label='$\sigma^2_y(t)$')
##        plt.plot(KORAKI, np.array(madnY)**2, 'b:x', label='$MADN^2_y(t)$')   
##        plt.title('Graf razmazanosti končnih leg ($ x_n $,$ y_n $) ob različnih časih sprehoda za ($\mu=$'+str(mu)+')')
##        plt.xlabel('t')
##        plt.ylabel('$\sigma^2$')
##        plt.xscale('log')
##        plt.yscale('log')   
##        plt.legend(loc='best')
##        
##    else:
##        #X,sigx,madnx,Y,sigy,madny,R,sigr,madnr=koncneTockeSFER(mu, 10000, sprehodi)
#    koraki=1000
#    X,sigx,madnx,Y,sigy,madny=koncneTockeSFER(mu, koraki, sprehodi)
#
#    GAMAsigx.append((2.*np.log(sigx)-np.log(sigX[0]**2))/(np.log(koraki)))  # vpišem gamma iz deviacije  po 10000 sprehodih
#    GAMAmadnx.append((2.*np.log(madnx)-np.log(madnX[0]**2))/(np.log(koraki))) # vpišem gamma iz robustne deviacije  po 10000 sprehodih
#    GAMAsigy.append((2.*np.log(sigy)-np.log(sigY[0]**2))/(np.log(koraki)))
#    GAMAmadny.append((2.*np.log(madny)-np.log(madnY[0]**2))/(np.log(koraki)))
##    
#FIG3= plt.figure() # izris grafa gamme za 10000 KORAKOV ob različnih mu
#plt.plot(MU, GAMAsigx, 'r--x', label='$\sigma_x(t)$')
#plt.plot(MU, GAMAmadnx, 'r:x', label='$MADN_x(t)$')
#plt.plot(MU, GAMAsigy, 'b--x', label='$\sigma_y(t)$')
#plt.plot(MU, GAMAmadny, 'b:x', label='$MADN_y(t)$')   
#plt.title(r'Potenca časovne odvisnosti ($t^{\gamma}$) razmazanosti končnih leg po 1000 korakih za različne parametre $\mu$')
#plt.ylabel('$\gamma$')
#plt.xlabel('$\mu$')
#plt.legend(loc='best')    
#plt.show()


##### izris histogramov porazdelitve končnih točk  v kartezičnih koordinatah 
sprehodi=10**4 # število sprehodov za dobro statistiko

for k in range (4,5):
    sprehodi=10**4
    funkcija=k
    FUNKCIA=['$x$','$|x|$','$1/x$','$1/|x|$','$x^{4}\ln|1+x|$']
    print(FUNKCIA[k])
    
    #X,sigx,madnx,Y,sigy,madny,R,sigr,madnr=koncneTockeKART(funkcija, 1, sprehodi)
    #sigX,sigY,sigR = [sigx],[sigy],[sigr]
    #madnX,madnY,madnR = [madnx],[madny],[madnr]
    X,sigx,madnx,Y,sigy,madny=koncneTockeKART(funkcija, 1, sprehodi)
    sigX,sigY = [sigx],[sigy]
    madnX,madnY = [madnx],[madny]
    KORAKI=[1,5,10,50,100,500,1000]  

    
    for l in range (1,7):#7
        koraki=KORAKI[l]
        
        #X,sigx,madnx,Y,sigy,madny,R,sigr,madnr=koncneTockeKART(funkcija, koraki, sprehodi)
        X,sigx,madnx,Y,sigy,madny=koncneTockeKART(funkcija, koraki, sprehodi)

        sigX.append(sigx)  # vpišem deviacije sprehodov
        sigY.append(sigy)
#            sigR.append(sigr)

        madnX.append(madnx)  # vpišem robustne deviacije sprehodov
        madnY.append(madny)
#            madnR.append(madnr)     
        
        if koraki==100:
            X10=X
            Y10=Y
            npx10=StStolp(X)
            npy10=StStolp(Y)
            
        elif koraki==500:
            X100=X
            Y100=Y
            npx100=StStolp(X)
            npy100=StStolp(Y)
    
    npx=StStolp(X)
    npy=StStolp(Y)

    FIG= plt.figure() # izris histogramov porazdelitve končnih točk za izaračun s 100,1000 in 10000 koraki     
    plt.hist(X10, npx10, normed=True, histtype='step', color='m', label='100 korakov $(x)$') #izris histograma
    plt.hist(Y10, npy10, normed=True, histtype='step', color='c', label='100 korakov $(y)$')#izris histograma
    plt.hist(X100, npx100, normed=True, histtype='step', color='r', label='500 korakov $(x)$') #izris histograma
    plt.hist(Y100, npy100, normed=True, histtype='step', color='b', label='500 korakov $(y)$')#izris histograma
    plt.hist(X, npx, normed=True, histtype='step', color='y', label='1000 korakov $(x)$') #izris histograma
    plt.hist(Y, npy, normed=True, histtype='step', color='g', label='1000 korakov $(y)$')#izris histograma
    #    npr=StStolp(R)
    #    plt.hist(R, npr, normed=False, histtype='step', color='k', label='koordinate r')#izris histograma
    plt.title('Histogram porazdelitve končnih vrednosti koordinat pri '+str(sprehodi)+' sprehodih za '+FUNKCIA[k])
    plt.xlabel('x_n')
    plt.ylabel('N')
    plt.legend(loc='best')
      
    FIG2= plt.figure() # izris grafa razmazanosti z daljšanjem časa/KORAKOI=[1,100,500,1000,2000,5000,10000]
    plt.plot(KORAKI, np.array(sigX)**2, 'r--x', label='$\sigma^2_x(t)$')
    plt.plot(KORAKI, np.array(madnX)**2, 'r:x', label='$MADN^2_x(t)$')
    plt.plot(KORAKI, np.array(sigY)**2, 'b--x', label='$\sigma^2_y(t)$')
    plt.plot(KORAKI, np.array(madnY)**2, 'b:x', label='$MADN^2_y(t)$')   
    plt.title('Graf razmazanosti končnih leg ($ x_n $,$ y_n $) ob različnih časih sprehoda za '+FUNKCIA[k])
    plt.xlabel('t')
    plt.ylabel('$\sigma^2$')
    plt.xscale('log')
    plt.yscale('log')   
    plt.legend(loc='best')

#    
