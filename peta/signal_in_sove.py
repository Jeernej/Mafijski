# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Wed Nov  2 13:32:41 2016

@author: jernej
"""
#import sys
from math import *
from matplotlib import *
from pylab import *


import pylab
import matplotlib.pyplot as plt
from matplotlib import ticker
#from scipy import fftpack
import numpy as np
import scipy.fftpack
from numpy import loadtxt
from numpy import savetxt
#from numpy import fft
#from scipy.special import airy, gamma
#from scipy import fftconvolve
#from scipy import convolve
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftfreq


#import matplotlib.animation as animation
#import time

#rfft(a[, n, axis, norm])     Compute the one-dimensional discrete Fourier Transform for real input.
#This function computes the one-dimensional n-point discrete Fourier Transform (DFT) 
#of a real-valued array by means of an efficient algorithm called the Fast Fourier Transform (FFT).

#irfft(a[, n, axis, norm])     Compute the inverse of the n-point DFT for real input.


##__________definicije_uporabljenih_funkcij________________________________________________________________________________________________

# scipy.signal.fftconvolve
# convolve(a, v, mode='full')
# signal.fftconvolve(sig, sig[::-1], mode='full')
# np.convolve([1, 2, 3], [0, 1, 0.5])


def Kor( SIG1, SIG2, varianta):
    
    LS1= len(SIG1) # dolžina  signala
    LS2= len(SIG2)

    if LS1<LS2:
        SIG2 = signal.resample(SIG2, LS1)
#
    elif LS1>LS2:
        SIG1 = signal.resample(SIG1, LS2)   #
    else:
        pass
        
    if varianta==0:
        KOR=signal.fftconvolve(SIG1, SIG2[::-1], mode='same')

    else:
        
        FT1 = fft(SIG1) 
        FT2 = np.ndarray.conjugate(fft(SIG2))
            
        KOR=ifft(FT1*(FT2))     
        
    return KOR


    
def AKor(SIG, varianta):

    if varianta==0:
        
        AKOR=signal.fftconvolve(SIG, SIG[::-1], mode='full') 

    else:
    
        FT = fft(SIG) 
        FT = np.ndarray.conjugate(fft(SIG))
            
        AKOR=ifft((abs(FT))**2)
        
    return AKOR
    
    
#                                       RAČUNANJE IN TABELIRANJE

DIR = '/home/jernej/Desktop/Mafijski Praktikum/MOJEDELLO/peta/' # pot do delovne mape s podatki

#    
#x = np.linspace(-8*np.pi, 8*np.pi, num=500)
#noise =np.random.normal(0, 1,500)
#pylab.figure()
#plt.subplot(311)
#plt.plot(noise)
#signal_krivulje = cos(4/7*x)+scipy.special.diric(x,11)
#plt.subplot(312)
#plt.plot(signal_krivulje)
#krivulje_random_noise=(signal_krivulje+noise)*noise
#plt.subplot(313)
#plt.plot(krivulje_random_noise)
#
#savetxt( DIR +"signal_krivulje", signal_krivulje)
#savetxt( DIR +"krivulje_random_noise", krivulje_random_noise)


generiran = ["signal_krivulje","random_noise","krivulje_random_noise"]

sove = ["bubomono","bubo2mono","mix","mix1","mix2","mix22"]

#                                             PARAMETRI:

'''
 V= številka za izbiro vzorca
 1 = sova1 = open(DIR + podatki[2],"r")
 2 = sova2 = open(DIR + podatki[2],"r")
 3 = mixCicada = open(DIR + podatki[3],"r")
 4 = mixPotok = open(DIR + podatki[4],"r")
 5 = mixReka = open(DIR + podatki[5],"r")
 6 = mixREKA = open(DIR + podatki[6],"r")
 7 = ligo = open(DIR + podatki[6],"r") # Hanford Strain Data at 16384 Hz, 32s   
 '''
 
podatki=generiran

vzorecA=0 # izbiraj med 0 in 6
vzorecB=1 # izbiraj med 0 in 6
#avtokorelacija=1 #  avtokorelacija = 1 
nacin=0 # moj način=1

# AVTOKORELACIJA    

if vzorecA == vzorecB:
    SIG = loadtxt(DIR + podatki[vzorecA]+".txt") # branje
    Lsig= len(SIG)
    
    AKOR_A= AKor(SIG, nacin) # izračun korelacij vrne AKOR !!    
    L= len(AKOR_A) # dolžina signala
    
#                            RISANJE PODATKOV
    pylab.figure()
    
    plt.subplot(121)
    plt.plot(SIG)
    pylab.xlim([0, Lsig])#območje vrednosti x na grafu
    pylab.title("signal:"+ podatki[vzorecA] )
    pylab.ylabel("apmlituda [arbitrarno]")
    pylab.xlabel("t [arbitrarno]")
    
    plt.subplot(122)
    plt.plot(AKOR_A)
    
    if nacin ==1:
        pylab.xlim([0, L/2])    #območje vrednosti x na grafu
    else:
        pylab.xlim([L/2, L])  
     
    pylab.title("avtokorelacija:"+ podatki[vzorecA] )
    pylab.ylabel("apmlituda [arbitrarno]")
    pylab.xlabel("t [arbitrarno]")
    

# KORELACIJA 

else:

    SIG1 = loadtxt(DIR + podatki[vzorecA]+".txt") # branje
    SIG2 = loadtxt(DIR + podatki[vzorecB]+".txt") # branje
    LS1= len(SIG1) # dolžina avtokoreliranega signala
    LS2= len(SIG2) # dolžina avtokoreliranega signala
    
    KOR_AB=Kor( SIG1, SIG2, nacin)
    L= len(KOR_AB) # dolžina avtokoreliranega signala

    pylab.figure()

    plt.subplot(131)
    plt.plot(SIG1)
    pylab.xlim([0, LS1])#območje vrednosti x na grafu
    pylab.title("signal:"+ podatki[vzorecA] )
    pylab.ylabel("apmlituda [arbitrarno]")
    pylab.xlabel("t [arbitrarno]")

    plt.subplot(132)
    plt.plot(SIG2)
    pylab.xlim([0, LS2])#območje vrednosti x na grafu
    pylab.title("signal:"+ podatki[vzorecB] )
    pylab.ylabel("apmlituda [arbitrarno]")
    pylab.xlabel("t [arbitrarno]")
    
    plt.subplot(133)
    plt.plot(KOR_AB)
    
    if nacin ==1:
        pylab.xlim([0, L/2])    #območje vrednosti x na grafu
    else:
        pylab.xlim([L/2, L])  
        
    pylab.title("korelacija:"+ podatki[vzorecA]+"_"+ podatki[vzorecB] )
    pylab.ylabel("apmlituda [arbitrarno]")
    pylab.xlabel("t [arbitrarno]")
    
#    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/sek))
#    S2plot.xaxis.set_major_formatter(ticks_x)
#    S2plot.set_xticks([])
#    S2plot.set_yticks([])
#                           