# -*- coding: utf-8 -*-
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
    
    
'''
Now let's filter the signal in the time domain, using bandpassing to reveal the signal
in the frequency band [40 , 300 Hz], and notching of spectral lines to remove those noise sources
from the data.
'''
def filter_rule(x,freq):
    if abs(freq)>250 or abs(freq)<35:
        return 0
    else:
        return x
    
#                                       RAČUNANJE IN TABELIRANJE

DIR = '/home/jernej/Desktop/Mafijski Praktikum/MOJEDELLO/peta/' # pot do delovne mape s podatki
primerjava = ["H1_4Hz_32s","L1_4Hz_32s","H1_16Hz_32s","L1_16Hz_32s","H1_preciscen", "L1_preciscen" ,"LIGO_SIG_template"]

podatki=primerjava


'''
The chirp signal lasted over 0.2 seconds, and increased in frequency and
amplitude in about 8 cycles from 35 Hz to 250 Hz.
The signal is visible as an oscillation sweeping from low to high frequency from -0.10 seconds to 0, 
then damping down into the random noise.
We had to shift the L1 data by 7 ms to get it to line up with the data from H1, 
because the source is roughly in the direction of the line connecting H1 to L1, 
and the wave travels at the speed of light, so it hits L1 7 ms earlier. 
Also, the orientation of L1 with respect to H1 means that we have to flip the sign of the signal in L1 
for it to match the signal in H1.
[https://losc.ligo.org/s/events/GW150914/GW150914_tutorial.html]
'''
#                                             PARAMETRI:


vzorecA=1 # izbiraj med 0 in 6
vzorecB=6 # izbiraj med 0 in 6
nacin=1 # izbiraj med 1=FFT in 0


fs=4096
#fs=16384
#fs=2769 # teoretična krivulja
fst=3250
dt=1/fs
dtt=1/fst

#
i=1000
korak=0.002
pot=0.2
zamik=-pot+i*korak
# AVTOKORELACIJA    
#for i in range(0, int(2*pot/korak)):
#    zamik=-pot+i*korak
if vzorecA == vzorecB:
    
    SIG = loadtxt(DIR + podatki[vzorecA]+".txt") # branje
    L= len(SIG) # dolžina signala
    platno = plt.figure() #izriše vse na en graf

    if podatki==primerjava:
        if vzorecA==0:
   
            sek=fs
            DT=ceil(sek*zamik)
            xs1=ceil(L/2-0.02*sek)+DT        
            xs2=ceil(L/2+0.18*sek)+DT
            
            plt.subplot(321) #izriše vse na en graf
            plt.plot(SIG)
            plt.xlim([0, L]) 
            pylab.axvline( xs1, linestyle=':',color='r', linewidth=1 )#pomožna črta prehoda AsNeg v MAc
            pylab.axvline( xs2, linestyle=':',color='r', linewidth=1 )#pomožna črta prehoda AsNeg v MAc

            
            SIG =SIG[xs1:xs2] # H1 podmnožica ki vsebuje signalž
            SIG=SIG[::-1] # reverse element order
            
            F1 = fft(SIG)          # take the fourier transform of the data
            f1 = fftfreq(len(F1),dt) # get sample frequency in cycles/sec (Hz)
            F1_filtered = array([filter_rule(x,freq) for x,freq in zip(F1,f1)]) # filter the Fourier transform
            SIG_filtered = ifft(F1_filtered) 
            
            LS_filtered= len(SIG_filtered) # dolžina avtokoreliranega signala
            plt.subplot(325) #izriše vse na en graf
            plt.plot(SIG_filtered)
            plt.xlim([0, LS_filtered]) 

        elif vzorecA==1:
   
            sek=fs
            DT=ceil(sek*zamik)
            xs1=ceil(L/2-0.13*sek)+DT        
            xs2=ceil(L/2+0.13*sek)+DT
            
            plt.subplot(321) #izriše vse na en graf
            plt.plot(SIG)
            plt.xlim([0, L]) 
            pylab.axvline( xs1, linestyle=':',color='r', linewidth=1 )#pomožna črta prehoda AsNeg v MAc
            pylab.axvline( xs2, linestyle=':',color='r', linewidth=1 )#pomožna črta prehoda AsNeg v MAc

            
            SIG =SIG[xs1:xs2] # H1 podmnožica ki vsebuje signalž
            SIG=SIG[::-1] # reverse element order
            
            F1 = fft(SIG)          # take the fourier transform of the data
            f1 = fftfreq(len(F1),dt) # get sample frequency in cycles/sec (Hz)
            F1_filtered = array([filter_rule(x,freq) for x,freq in zip(F1,f1)]) # filter the Fourier transform
            SIG_filtered = ifft(F1_filtered) 
            
            LS_filtered= len(SIG_filtered) # dolžina avtokoreliranega signala
            plt.subplot(325) #izriše vse na en graf
            plt.plot(SIG_filtered)
            plt.xlim([0, LS_filtered]) 
            
        elif vzorecA==6:
            SIG =SIG[1850:2695]

        else:
            pass # dummy statment does nothing

        LS= len(SIG) # dolžina avtokoreliranega signala
        plt.subplot(323) #izriše vse na en graf
        plt.plot(SIG)
        plt.xlim([0, LS]) 
        
        
    else:
        pass # dummy statment does nothing
       
    AKOR_A= AKor(SIG, nacin) # izračun korelacij vrne AKOR !!    
#    savetxt( DIR +"AKOR_dx"+ str(zamik) +"_"+ podatki[vzorecA], AKOR_A )
    L= len(AKOR_A)
#                            RISANJE PODATKOV
    plt.subplot(324) #izriše vse na en graf
    pylab.plot(AKOR_A)
    if nacin ==1:
        pylab.xlim([0, L/2])    #območje vrednosti x na grafu
    else:
        pylab.xlim([L/2, L])     
    pylab.title("signal"+ podatki[vzorecA] )
    pylab.ylabel("apmlituda [arbitrarno]")
    pylab.xlabel("t [arbitrarno]")
    
    if vzorecA==0 or vzorecA==1:
        AKOR_A_filtered= AKor(SIG_filtered, nacin) # izračun korelacij vrne AKOR !!    
        L_filtered= len(AKOR_A_filtered)
        plt.subplot(326) #izriše vse na en graf
        pylab.plot(AKOR_A_filtered)
        if nacin ==1:
            pylab.xlim([0, L_filtered/2])    #območje vrednosti x na grafu
        else:
            pylab.xlim([L_filtered/2,L_filtered])          
        pylab.title("signal"+ podatki[vzorecA] )
        pylab.ylabel("apmlituda [arbitrarno]")
        pylab.xlabel("t [arbitrarno]")
        
    else:
        pass
    
#    vrednosti = loadtxt(DIR +"AKOR_dx"+ str(zamik) + "_" + podatki[vzorecA] )


# KORELACIJA 

#def animate(frame):
#    zamik=-1+frame*0.1  # izbiraj zamik za avtokorelacijo signala
else:

    SIG1 = loadtxt(DIR + podatki[vzorecA]+".txt") # branje
    SIG2 = loadtxt(DIR + podatki[vzorecB]+".txt") # branje
    LS1= len(SIG1) # dolžina avtokoreliranega signala
    LS2= len(SIG2) # dolžina avtokoreliranega signala
    platno = plt.figure() #izriše vse na en graf
    plt.subplot(431) #izriše vse na en graf
    plt.plot(SIG1)
    plt.xlim([0, LS1]) 
    plt.subplot(432) #izriše vse na en graf
    plt.plot(SIG2)
    plt.xlim([0, LS2]) 
#
#    if podatki==LIGO:
#        sek=fs
#        DT=int(ceil(sek*zamik))             
#        SIG1 =SIG1[ceil(LS2/2-0.065*sek)+DT:ceil(LS2/2+0.215*sek)+DT] # podmnožica ki vsebuje signal
#        SIG1=SIG1[::-1] # reverse element order
#        SIG2 =SIG2[ceil(LS2/2-0.175*sek):ceil(LS2/2+0.16*sek)]
#        SIG2=SIG2[::-1] # reverse element order
   
    if podatki==primerjava:
        if vzorecA==0 or vzorecA==2:
            sek=fs
            DT=int(ceil(sek*zamik)) 
            SIG1 =SIG1[ceil(LS1/2-0.02*sek)+DT:ceil(LS1/2+0.18*sek)+DT] # H1 podmnožica ki vsebuje signal
            SIG1=SIG1[::-1] # reverse element order
            if vzorecB==6:
                SIG2 =SIG2[1850:2695] # podmnožica signala
            else:
                pass
            LS1= len(SIG1) # dolžina avtokoreliranega signala
            LS2= len(SIG2)
            plt.subplot(434) #izriše vse na en graf
            plt.plot(SIG1)
            plt.xlim([0, LS1])
            plt.subplot(435) #izriše vse na en graf
            plt.plot(SIG2)
            plt.xlim([0, LS2])
        elif vzorecA==1 or vzorecA==3:
            sek=fs
            DT=int(ceil(sek*zamik)) 
            SIG1 =SIG1[ceil(LS1/2-0.13*sek)+DT:ceil(LS1/2+0.13*sek)+DT] # L1 podmnožica ki vsebuje signal
            SIG1=SIG1[::-1] # reverse element order
            if vzorecB==6:
                SIG2 =SIG2[1850:2695] # podmnožica signala
            else:
                pass
            LS1= len(SIG1) # dolžina avtokoreliranega signala
            LS2= len(SIG2)
            plt.subplot(434) #izriše vse na en graf
            plt.plot(SIG1)
            plt.xlim([0, LS1])
            plt.subplot(435) #izriše vse na en graf
            plt.plot(SIG2)
            plt.xlim([0, LS2])
            
        elif vzorecB==6:
            SIG2 =SIG2[1850:2695] # podmnožica signala
        else:
            pass # dummy statment does nothing
        
        LS1= len(SIG1) # dolžina avtokoreliranega signala
        LS2= len(SIG2) # dolžina avtokoreliranega signala
        
      
    else:
        pass # dummy statment does nothing
#    
    KOR_AB=Kor(SIG2, SIG1, nacin) # izračun korelacij vrne KOR !!
    L= len(KOR_AB) # dolžina koreliranega signala

    plt.subplot(436) #izriše vse na en graf
    plt.plot(KOR_AB)
#    plt.xlim([0, L])
    if nacin ==1:
        pylab.xlim([0, L/2])    #območje vrednosti x na grafu
    else:
        pylab.xlim([L/2, L])
        
    '''
    Now let's filter the signal in the time domain, using bandpassing to reveal the signal
    in the frequency band [40 , 300 Hz], and notching of spectral lines to remove those noise sources
    from the data.
    '''
    F1 = fft(SIG1)          # take the fourier transform of the data
    F2t = fft(SIG2)          # take the fourier transform of the data
    f1 = fftfreq(len(F1),dt) # get sample frequency in cycles/sec (Hz)
    f2t = fftfreq(len(F2t),dtt) # get sample frequency in cycles/sec (Hz)

    def filter_rule(x,freq):
        if abs(freq)>250 or abs(freq)<35:
            return 0
        else:
            return x
    
    F1_filtered = array([filter_rule(x,freq) for x,freq in zip(F1,f1)]) # filter the Fourier transform
    
    SIG1_filtered = ifft(F1_filtered) # reconstruct the filtered signal
#    
    if LS1<LS2:
        SIG2 = signal.resample(SIG2, LS1)   #frequency 250
#        SIG2_filtered = signal.resample(SIG2_filtered, LS1)   #frequency 250

    else:
        SIG1 = signal.resample(SIG1, LS2)   #frequency 250
        SIG1_filtered = signal.resample(SIG1_filtered, LS2)   #frequency 250

    
    LS1= len(SIG1) # dolžina avtokoreliranega signala
    LS2= len(SIG2)
    LS1_F= len(SIG1_filtered) # dolžina avtokoreliranega signala

    KOR_AB_resamp=Kor(SIG2, SIG1, nacin) # izračun korelacij vrne KOR !!
    KOR_AB__filtered=Kor(SIG2, SIG1_filtered, nacin) # izračun korelacij vrne KOR !!

    L_resamp= len(KOR_AB) # dolžina koreliranega signala
    L_F= len(KOR_AB__filtered) # dolžina koreliranega signala
        
    plt.subplot(437) #izriše vse na en graf
    plt.plot(SIG1)
    plt.xlim([0, LS1])
    plt.subplot(438) #izriše vse na en graf
    plt.plot(SIG2)
    plt.xlim([0, LS2])
    plt.subplot(439) #izriše vse na en graf
    plt.plot(KOR_AB_resamp)
    if nacin ==1:
        pylab.xlim([0, L_resamp/2])    #območje vrednosti x na grafu
    else:
        pylab.xlim([L_resamp/2, L_resamp])
        
    plt.subplot(4,3,10) #izriše vse na en graf
    plt.plot(SIG1_filtered)
    plt.xlim([0, LS1])
    plt.subplot(4,3,11) #izriše vse na en graf
    plt.plot(SIG2)
    plt.xlim([0, LS2])
    plt.subplot(4,3,12) #izriše vse na en graf
    plt.plot(KOR_AB__filtered)
    if nacin ==1:
        pylab.xlim([0, L_F/2])    #območje vrednosti x na grafu
    else:
        pylab.xlim([L_F/2, L_F]) 
    
#    savetxt( DIR +"KOR_" + podatki[vzorecA] + podatki[vzorecB], KOR_AB)
    
#    return KOR_AB, SIG1, SIG2
    
#    
#    
#def init():
#    SIG1=[]
#    SIG2=[]
#    KOR_AB=[]
#    return KOR_AB, SIG1, SIG2, 


#                            RISANJE PODATKOV
    #vrednosti =  loadtxt(DIR +"KOR_" + podatki[vzorecA]+ podatki[vzorecB])
    #L= len(KOR_AB,) # dolžina koreliranega signala
#    platno = plt.figure() #izriše vse na en graf
#    #-----
#    KORplt = plt.subplot(313) #izriše vse na en graf
#    plt.plot(KOR_AB)
##    plt.plot(KOR_AB__filtered)
#    #plt.psd(SIG1,SIG2,fs,fs)
#    #KOR_AB,=KORplt.plot([])
#    pylab.xlim([0, L]) 
#    plt.title("korelacija:"+ podatki[vzorecA] +"~"+ podatki[vzorecB])
#    #plt.ylabel("apmlituda [arbitrarno]")
#    #plt.xlabel("t [arbitrarno]")
#    KORplt.set_xticks([])
#    KORplt.set_yticks([])
##    -----
##    S1Fplot=plt.subplot(313) #izriše vse na en graf
##    plt.plot(SIG1_filtered)
###    plt.plot(SIG1_filtered)
##    #plt.psd(SIG1,fs,fs)
##    #SIG1,=S1plot.plot([])
##    plt.xlim([0, LS1]) 
##    plt.title("signal:"+ podatki[vzorecA])
##    #plt.ylabel("apmlituda [arbitrarno]")
##    #plt.xlabel("t [arbitrarno]")
##    #ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/sek))
##    #S1plot.xaxis.set_major_formatter(ticks_x)
##    S1Fplot.set_xticks([])
##    #S1plot.set_yticks([])
#    #-----
#    S1plot=plt.subplot(312) #izriše vse na en graf
#    plt.plot(SIG1)
##    plt.plot(SIG1_filtered)
##    plt.plot(SIG1_filtered)
#    #plt.psd(SIG1,fs,fs)
#    #SIG1,=S1plot.plot([])
#    plt.xlim([0, LS1]) 
#    plt.title("signal:"+ podatki[vzorecA])
#    #plt.ylabel("apmlituda [arbitrarno]")
#    #plt.xlabel("t [arbitrarno]")
#    #ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/sek))
#    #S1plot.xaxis.set_major_formatter(ticks_x)
#    S1plot.set_xticks([])
#    #S1plot.set_yticks([])
#    #-----
#    S2plot=plt.subplot(311) #izriše vse na en graf
##    plt.plot(F2t)
#    plt.plot(SIG2)
#    #plt.psd(SIG2,fs,fs)
#    #SIG2,=S2plot.plot([])
#    plt.xlim([0, LS2]) 
#    plt.title("signal:"+ podatki[vzorecB])
#    #plt.ylabel("apmlituda [arbitrarno]")
#    #plt.xlabel("t [arbitrarno]")
#    #ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/sek))
#    #S2plot.xaxis.set_major_formatter(ticks_x)
#    S2plot.set_xticks([])
#    #S2plot.set_yticks([])

#-----

#frame=int(2*pot/korak)
#anim = animation.FuncAnimation(platno, animate,  frames=20, interval=20) #init_func=init,
#anim.save(DIR+'double_pendulum.mp4', fps=2)

  