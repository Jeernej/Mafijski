## -*- coding: utf-8 -*-

import scipy as sc
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

def detect_local_minima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html #binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min - eroded_background
    return np.where(detected_minima)     
    


#_____ RK4 adaptive step rešitev________________________________________________


def control(ddxt, tol, xt0, dxt0, t0, tmax, v0, frekvenca): #, v=False):
    '''
#    This funciton takes in a python function that returns the derivative,
#    a tolerance for error between RK5 and RK4, initial conditions on y (dependant) 
#    and t (independant) as well as an initial step size.   
    '''
    h    = 0.01           # Initial step size

    Xt = [xt0]      # Set up the initial conditions on x(t)
    dXt = [dxt0]      # Set up the initial conditions on dx(t)
    energijaRK4 = [1-np.cos(xt0)+(dxt0)**2/2]      # Set up the initial conditions on energy

    t = [t0]      # and t while creating the output lists
    H = [h]   
   
    t_curr = t0
    xt_curr = xt0 # Setup counters and trackers
    dxt_curr =  dxt0  # Setup counters and trackers

    while t_curr < tmax:
        
        dxt_next, h, h2 = stepper(ddxt, xt_curr, dxt_curr, h, tol, t_curr, v0, frekvenca, optimizacija=True) # pridobi optimalni korak  h2
        
        xt_next, h, h1 = stepper2( xt_curr, dxt_curr, h, h, tol, t_curr, v0, frekvenca, optimizacija=True)  # pridobi optimalni korak h1
        

        
        if h1 < h2: # uporabi krajši optimalni korak od obeh in računaj z njim šeenkrat obe

            h = h1
            dxt_next, h , h2  = stepper(ddxt, xt_curr, dxt_curr, h, tol, t_curr, v0, frekvenca, optimizacija=False)
            xt_next, h, h1 = stepper2( xt_curr, dxt_curr, h, h, tol, t_curr, v0, frekvenca, optimizacija=False)

            Xt.append(xt_next)
            dXt.append(dxt_next)
            energijaRK4.append(1-np.cos(xt_next)+(dxt_next)**2/2)
            t.append(t_curr+h)
            H.append(h)
            xt_curr = xt_next
            dxt_curr= dxt_next
            t_curr = t_curr+h
            
#            print(h1)
#            print(h2)
            
        elif h1 > h2: # uporabi krajši optimalni korak od obeh in računaj z njim šeenkrat obe

            h = h2

            dxt_next, h, h2  = stepper(ddxt, xt_curr, dxt_curr, h, tol, t_curr, v0, frekvenca, optimizacija=False)
            xt_next, h, h1 = stepper2( xt_curr, dxt_curr, h, h, tol, t_curr, v0, frekvenca, optimizacija=False)

            Xt.append(xt_next)
            dXt.append(dxt_next)
            energijaRK4.append(1-np.cos(xt_next)+(dxt_next)**2/2)
            t.append(t_curr+h)
            H.append(h)
            xt_curr = xt_next
            dxt_curr= dxt_next
            t_curr = t_curr+h
            
#            print('+'+str(h1))
#            print('+'+ str(h2))
            
        else:  # če sta optimalna koraka enaka računaj z njim šeenkrat obe

            h = h1

            dxt_next, h, h2  = stepper(ddxt, xt_curr, dxt_curr, h, tol, t_curr, v0, frekvenca, optimizacija=False)
            xt_next, h, h1  = stepper2( xt_curr, dxt_curr, h, h, tol, t_curr, v0, frekvenca, optimizacija=False)
            
            Xt.append(xt_next)
            dXt.append(dxt_next)
            energijaRK4.append(1-np.cos(xt_next)+(dxt_next)**2/2)
            t.append(t_curr+h)
            H.append(h)
            xt_curr = xt_next
            dxt_curr= dxt_next
            t_curr = t_curr+h
            
    t=np.array(t)
    return np.array(Xt), np.array(dXt), energijaRK4, np.array(t) ,H
   
   
a2,   a3,  a4,  a5,  a6      =        1/5.,    3/10.,       3/5.,            1.,        7/8.
b21, b31, b32, b41, b42, b43 =        1/5.,    3/40.,      9/40.,         3/10.,      -9/10., 6/5.
b51, b52, b53, b54           =     -11/54.,     5/2.,    -70/27.,        35/27.
b61, b62, b63, b64, b65      = 1631/55296., 175/512., 575/13824., 44275/110592.,   253/4096. 
c1,   c2,  c3,  c4,  c5, c6  =     37/378.,       0.,   250/621.,      125/594.,          0.,  512/1771.
c1star, c2star, c3star, c4star, c5star, c6star = 2825/27648., 0.,  18575/48384.,13525/55296., 277/14336., 1/4.
    
def stepper(deriv, t, y, h, tol, T, v0, frekvenca, optimizacija):
    '''
#   This function is called by the control function to take a single step forward. The inputs are the derivative function,
#   the previous time and function value, the step size in time (h), and the tolerance
#   for error between 5th order Runge-Kutta and 4th order Runge-Kutta.
    '''
    k1 = h*deriv(t,y, T, v0, frekvenca)
    k2 = h*deriv(t+a2*h,y+b21*k1, T, v0, frekvenca)
    k3 = h*deriv(t+a3*h,y+b31*k1+b32*k2, T, v0, frekvenca)
    k4 = h*deriv(t+a4*h,y+b41*k1+b42*k2+b43*k3, T, v0, frekvenca)
    k5 = h*deriv(t+a5*h,y+b51*k1+b52*k2+b53*k3+b54*k4, T, v0, frekvenca)
    k6 = h*deriv(t+a6*h,y+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5, T, v0, frekvenca)  # za izračun z natančnejšo metodo (reda 5) za izbiro koraka
   
    y_n_plus_1      = y +     c1*k1 +     c2*k2 +     c3*k3 +     c4*k4 +     c5*k5 +     c6*k6
    y_n_plus_1_star = y + c1star*k1 + c2star*k2 + c3star*k3 + c4star*k4 + c5star*k5 + c6star*k6
   
    DELTA           = y_n_plus_1 - y_n_plus_1_star
    
    h1 = h   
          
    if optimizacija==True:
        try:
            h1 = h*abs(tol/DELTA)**0.2    # Finds step size required to meet given tolerance 
#            print(h1)
        except ZeroDivisionError:
            
            h1 = h                        # When you are very close to ideal step, DELTA can be zero
            
            k1 = h*deriv(t,y, T, v0, frekvenca)
            k2 = h*deriv(t+a2*h,y+b21*k1, T, v0, frekvenca)
            k3 = h*deriv(t+a3*h,y+b31*k1+b32*k2, T, v0, frekvenca)
            k4 = h*deriv(t+a4*h,y+b41*k1+b42*k2+b43*k3, T, v0, frekvenca)
            k5 = h*deriv(t+a5*h,y+b51*k1+b52*k2+b53*k3+b54*k4, T, v0, frekvenca)
            k6 = h*deriv(t+a6*h,y+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5, T, v0, frekvenca)
            y_n_plus_1      = y +     c1*k1 +     c2*k2 +     c3*k3 +     c4*k4 +     c5*k5 +     c6*k6
#            print(h1)
    else:
        pass
#    h1 = h                       
#    print(h1)
    return y_n_plus_1, h, h1

def stepper2( xt_curr, dxt_curr, h, H, tol, T, v0, frekvenca, optimizacija):
    '''
#   This function is called by the control function to take a single step forward. The inputs are the derivative function,
#   the previous time and function value, the step size in time (h), and the tolerance
#   for error between 5th order Runge-Kutta and 4th order Runge-Kutta.
    '''
    
    k1 = h*dxt_curr
    k2 = h*dxt( ddxt, xt_curr+a2*h, dxt_curr+b21*k1, H, tol , T, v0, frekvenca)
    k3 = h*dxt( ddxt, xt_curr+a3*h, dxt_curr+b31*k1+b32*k2, H, tol , T, v0, frekvenca)
    k4 = h*dxt( ddxt, xt_curr+a4*h, dxt_curr+b41*k1+b42*k2+b43*k3, H, tol , T, v0, frekvenca)
    k5 = h*dxt( ddxt, xt_curr+a5*h, dxt_curr+b51*k1+b52*k2+b53*k3+b54*k4, H, tol , T, v0, frekvenca)
    k6 = h*dxt( ddxt, xt_curr+a6*h, dxt_curr+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5, H, tol , T, v0, frekvenca) # za izračun z natančnejšo metodo (reda 5) za izbiro koraka
    
    y_n_plus_1      = xt_curr +     c1*k1 +     c2*k2 +     c3*k3 +     c4*k4 +     c5*k5 +     c6*k6
    y_n_plus_1_star = xt_curr + c1star*k1 + c2star*k2 + c3star*k3 + c4star*k4 + c5star*k5 + c6star*k6
   
    DELTA           = y_n_plus_1 - y_n_plus_1_star
    
    h1 = h      
                  # When you are very close to ideal step, DELTA can be zero
    if optimizacija==True:
        try:
            h1 = h*abs(tol/DELTA)**0.2    # Finds step size required to meet given tolerance
    
    #            print(h1)
        except ZeroDivisionError:
            h1 = h                        # When you are very close to ideal step, DELTA can be zero
            
            k1 = h* dxt_curr
            k2 = h*dxt( ddxt, xt_curr+a2*h, dxt_curr+b21*k1, H, tol , T, v0, frekvenca)
            k3 = h*dxt( ddxt, xt_curr+a3*h, dxt_curr+b31*k1+b32*k2, H, tol , T, v0, frekvenca)
            k4 = h*dxt( ddxt, xt_curr+a4*h, dxt_curr+b41*k1+b42*k2+b43*k3, H, tol , T, v0, frekvenca)
            k5 = h*dxt( ddxt, xt_curr+a5*h, dxt_curr+b51*k1+b52*k2+b53*k3+b54*k4, H, tol , T, v0, frekvenca)
            k6 = h*dxt( ddxt, xt_curr+a6*h, dxt_curr+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5, H, tol , T, v0, frekvenca)
            y_n_plus_1      = xt_curr +     c1*k1 +     c2*k2 +     c3*k3 +     c4*k4 +     c5*k5 +     c6*k6
#            print(h1)
    else:
        pass
        
    return y_n_plus_1, h, h1
    

#_____ analitična rešitev________________________________________________

#def Anlt(t,xt0):#,dxt0): # podam začetni približek in array  tistih časovnih točk, v katerih s posamezno metodo računam
#    w0=1.
#    Anx=[]
#    Andx=[]
#    E=[] 
#
#    c0=sc.sin(xt0*0.5)
#    K=sc.special.ellipk(c0*c0) # za xt0=1.
##    zamik=1.*10**(-0)
#    zamik=2*0.0048#0.001031259429999487#+0.007772213319996002
#    t = t+K#-zamik  # zamaknem ker analitična metoda v času t0 ne začne računati pri največji amplitudo in hitrosti 0
#                    # Zaradi računanja napake metod v istih točkah (glej izračuni)
#    for j in range(0,len(t)):
#        
#        sn,cn,dn,ph =sc.special.ellipj(K-w0*t[j], c0**2.)
#        anx=2*np.arcsin(c0*cn) # odmik
#        Anx.append(anx)       
#        
#        if j>0 and Anx[j-1] < Anx[j]: # ko se odmik veča je odvod pozitiven       
#            andx=np.sqrt(2*(np.cos(anx)-np.cos(xt0))* w0**2) 
#        else:
#            andx=-np.sqrt(2*(np.cos(anx)-np.cos(xt0))* w0**2)  # drugače je odvod negativen    
#            
##        if j>4 and Andx[j-3]<Andx[j-2]>Andx[j-1] :
##            print(t[j-2])
##        elif j>4 and Andx[j-3]>Andx[j-2]<Andx[j-1] :
##            print(t[j-2])
#            
#        Andx.append(andx)        
#        energija=1-np.cos(anx)+((andx)**2)/2
#        E.append(energija)        
#    return np.array(Anx), np.array(Andx), E



#_________________________ FUNKCIJE _________________________________________________

# Za reševanje s SCipy
def nihalo(y, t):
    
#    v0=0.8
#    frekvenca=2./3
    b = 0.5
    c = 1.
    xt, dxt = y
    
    ddxt = [dxt, -b*dxt - c*np.sin(xt)+ v0 *np.cos(frekvenca*t)]
    return ddxt

# Za reševanje z ostalimi metodami

def ddxt(xt, dxt, t, v0, frekvenca):
    b = 0.5
    c = 1.
    ddxt=-b*dxt - c*np.sin(xt)+ v0 *np.cos(frekvenca*t)
    return ddxt 

def dxt( ddxt, xt_curr, dxt_curr, H, tol , t, v0, frekvenca):
    dxt_next, h, h2 = stepper(ddxt, xt_curr, dxt_curr, H, tol, t, v0, frekvenca, optimizacija=True)
    return dxt_next

#_________________________ PARAMETRI _________________________________________________

pi=3.1415926535897932384626433

v0=0.8
frekvenca=2./3.
t0=0.               # Initial conditions
xt0  = 1.#pi/2.        # Initial conditions     
dxt0 = 0.           # Initial conditions
tmax = 20.*pi       # DOLŽINA INTERVALA      

t = np.linspace(0, tmax , 30*pi*1000)  # INTERVAL enakomerni korak      
t = np.array(t) 
#_________________________ RAČUNANJE _________________________________________________

# izračun RK4 adaptive

#tolRK  = 10**(-4.)     # The desired error between 4th and 5th order
                     # note that this IS NOT the error between numeric solution and the actual solution
#Xt, dXt, energijaRK4, trk4 ,dH = control( ddxt, tolRK, xt0, dxt0, t0, tmax, v0, frekvenca) #, v=True)
#print('RK4 konc')
#
## izračun SCIPY 
#
#pribl = [xt0, dxt0] # začetna pogoja  [xt0,dxt0]
#sol = odeint(nihalo, pribl, t) # začetna 
#Esci=1-np.cos(sol[:, 0])+(sol[:, 1])**2/2
#print('SCIPY konc')

# izračun trapezna
#
#hTr = 0.001           # Initial step size
#tx,Trx, tdx, Trdx, E = trapezna(hTr,tmax,xt0,dxt0)
#print('trapezna konc')
#
## izračun implicitna trapezna
#
#hTrIm = 0.001           # Initial step size
#tol= 10**(-14.)  
#Itx,TrIx, Itdx, TrIdx, IE = Trapezna_impl(hTrIm,tmax,xt0,dxt0,tol)
#print('implicitna trapezna konc')
#
## izračun analitična za enakomeren korak
#
#Anx, Andx, Ean =  Anlt(t,xt0)#,dxt0)
#
## izračun analitična za hitrosti trapeznih ki imajo enakomeren korak zamaknjen za h/2
#
#TAnx, TAndx, TEan =  Anlt(tdx,xt0)#,dxt0)

# izračun analitična za prilagojen korak RK4
#
#Anxrk4, Andxrk4, Erk4 =  Anlt(trk4,xt0)#,dxt0)
#print('analitične konc')
#
#local_minima_locations = detect_local_minima(-Anx)
#print(local_minima_locations)

#_________________________ RAČUNANJE in GRAFIRANJE za RAZLIČNE v0___________________________________________

#from matplotlib import gridspec
#
#dV=np.array([pi/6.,pi/5.,pi/4.,pi/3.,pi/2.]) # with 4 line syles '-','--','.-',':', dashes=[12,6,12,6,3,6]
#
#FIGampl= plt.figure()
#gs = gridspec.GridSpec(3, 1)
#odmiki = FIGampl.add_subplot(gs[0,0])
#hitrosti = FIGampl.add_subplot(gs[1,0])
#korakanja = FIGampl.add_subplot(gs[2,0])
#
#FIGfazni= plt.figure()
#gs2 = gridspec.GridSpec(3, 1)
#portreti = FIGfazni.add_subplot(gs2[:-1,0])
#energije = FIGfazni.add_subplot(gs2[2,0])
#
#for j in range(0,len(dV)):    
#    Xt, dXt, energijaRK4, trk4 ,dH = control( ddxt, tolRK, xt0, dxt0, t0, tmax, dV[j], frekvenca) #, v=True)    
##    rk4fig1.subplot(2, 1, 2 ) 
#    if j==0:
#        odmiki.plot(trk4, Xt, 'b:', label='pi/6')
#    elif j==1:
#        odmiki.plot(trk4, Xt, 'b-.', label='pi/5')
#    elif j==2:
#        odmiki.plot(trk4, Xt, 'b--', label='pi/4')
#    elif j==3:
#        odmiki.plot(trk4, Xt, dashes=[12,6,12,6,3,6], c='b', label='pi/3')
#    elif j==4:
#        odmiki.plot(trk4, Xt, 'b', label='pi/2')
##        odmiki.set_legend(loc='best')
#        odmiki.set_xlabel( '$t$' )
#        odmiki.set_ylabel( '$x(t)$' )
#        odmiki.set_title( 'Odmiki nihala računani s prilagodljivo dolžino koraka $\t{RK4}$')# (h='+str(korak)+')' )
#
##    rk4fig1.subplot(2, 1, 2 ) 
#    if j==0:
#        hitrosti.plot(trk4, dXt, 'g:', label='pi/6')
#    elif j==1:
#        hitrosti.plot(trk4, dXt, 'g-.', label='pi/5')
#    elif j==2:
#        hitrosti.plot(trk4, dXt, 'g--', label='pi/4')
#    elif j==3:
#        hitrosti.plot(trk4, dXt, dashes=[12,6,12,6,3,6], c='g', label='pi/3')
#    elif j==4:
#        hitrosti.plot(trk4, dXt, 'g', label='pi/2')        
##        hitrosti.legend(loc='best')
#        hitrosti.set_xlabel( '$t$' )
#        hitrosti.set_ylabel( '$v(t)$' )
#        hitrosti.set_title( 'Hitrosti nihala računane s prilagodljivo dolžino koraka $\t{RK4}$')# (h='+str(korak)+')' )
## 
###    rk4fig3=plt.subplot(1, 2, 1 )
##    if j==0:
##        korakanja.plot(trk4, dH, 'k:', label='pi/6')
##    elif j==1:
##        korakanja.plot(trk4, dH, 'k-.', label='pi/5')
##    elif j==2:
##        korakanja.plot(trk4, dH, 'k--', label='pi/4')
##    elif j==3:
##        korakanja.plot(trk4, dH, dashes=[12,6,12,6,3,6], c='k', label='pi/3')
##    elif j==4:
##        korakanja.plot(trk4, dH, 'k', label='pi/2')   
###        korakanja.legend(loc='best',bbox_to_anchor=[0, 0])
##        korakanja.set_xlabel( 'korak' )
##        korakanja.set_ylabel( 'dolžina koraka' )
##        korakanja.set_title( 'Spreminjanje dolžine koraka tekom iteracije pri računanju z metodo $\t{RK4}$')# (h='+str(korak)+')' ) 
##    
###    rk4fig2=plt.subplot(2, 2, 2 )
##    if j==0:
##        portreti.plot(Xt, dXt, 'k:', label='pi/6')
##    elif j==1:
##        portreti.plot(Xt, dXt, 'r-.', label='pi/5')
##    elif j==2:
##        portreti.plot(Xt, dXt, 'm--', label='pi/4')
##    elif j==3:
##        portreti.plot(Xt, dXt, dashes=[12,6,12,6,3,6], c='c', label='pi/3')
##    elif j==4:
##        portreti.plot(Xt, dXt, 'k', label='pi/2')        
###        portreti.legend(loc='best',bbox_to_anchor=[0, 0])
##        portreti.set_xlabel( '$x(t)$' )
##        portreti.set_ylabel( '$v(t)$' )
##        portreti.set_title( 'Fazni portreti' )
##    
##    scpyfig4=plt.subplot(2, 2, 4 )
##    if j==0:
##        energije.plot(trk4, energijaRK4, 'k:', label='pi/6')
##    elif j==1:
##        energije.plot(trk4, energijaRK4, 'r-.', label='pi/5')
##    elif j==2:
##        energije.plot(trk4, energijaRK4, 'm--', label='pi/4')
##    elif j==3:
##        energije.plot(trk4, energijaRK4, dashes=[12,6,12,6,3,6], c='c', label='pi/3')
##    elif j==4:
##        energije.plot(trk4, energijaRK4, 'k', label='pi/2') 
###        energije.legend(loc='best')
##        energije.set_ylabel('E(x(t),v(t))')
##        energije.set_xlabel('t')
##        energije.set_title( 'Energija nihala tekom iteracije pri računanju z metodo $\t{RK4}$')# (h='+str(korak)+')' ) 
#
#odmiki.legend(loc='upper right')
#hitrosti.legend(loc='upper right')
##korakanja.legend(loc='upper right')
#
##portreti.legend(loc='upper right')
##energije.legend(loc='upper right')
#


#_________________________ RAČUNANJE in GRAFIRANJE__HISTEREZE__za RAZLIČNE v0___________________________________________

def histereza( ddxt, tolRK, xt0, dxt0, t0, tmax, v0, frekvenca):
    
    frekvenca = 0.1 #začetna vrednost frekvence
    wmax=0.9 # najvišja vrednost frekvence
    wmin=0.5 # najvišja vrednost frekvence

    delta=0.05 # korak vrednost frekvence
    t0=0
    dT=tmax
    HISTampUP=[]
    HISTfrekUP=[]
         
    while frekvenca < wmax: 

        Xt, dXt, energijaRK4, trk4 ,dH = control( ddxt, tolRK, xt0, dxt0, t0, tmax, v0, frekvenca)   
        l=len(Xt)
        xt0=Xt[l-1]
        dxt0=dXt[l-1] 
        t0 =tmax # novi začetni pogoji - ob povišnju frekvence računam od tam kjer sem prej končal
        tmax=tmax+dT
        HISTampUP.append(abs(np.amax(Xt[int(l*4/5):])-np.amin(Xt[int(l*4/5):])))
        print(HISTampUP)
        HISTfrekUP.append(frekvenca)
        frekvenca = frekvenca + delta # višanje frekvence
        
    HISTampDWN=[]
    HISTfrekDWN=[]
    
    while frekvenca > wmin: 

        Xt, dXt, energijaRK4, trk4 ,dH = control( ddxt, tolRK, xt0, dxt0, t0, tmax, v0, frekvenca)        
        l=len(Xt)
        xt0=Xt[l-1]
        dxt0=dXt[l-1] 
        t0 =tmax # novi začetni pogoji - ob povišnju frekvence računam od tam kjer sem prej končal
        tmax=tmax+dT

        HISTampDWN.append(abs(np.amax(Xt[int(l*3/4):])-np.amin(Xt[int(l*3/4):])))
        print(HISTampDWN)

        HISTfrekDWN.append(frekvenca) 
        frekvenca = frekvenca - delta # nižanje frekvence
        
    return HISTampUP, HISTfrekUP, HISTampDWN, HISTfrekDWN,
           
  
 #__________________________računanje un izris____________________________________________   

dV=np.array([pi/10.,pi/7.,pi/5.,pi/4.,pi/3.]) # with 4 line syles '-','--','.-',':', dashes=[12,6,12,6,3,6]

FIGHist= plt.figure()
odmiki=plt.subplot(1, 1, 1 )

for j in range(0,len(dV)):   
    
    HISTampUP, HISTfrekUP, HISTampDWN, HISTfrekDWN =histereza( ddxt, tolRK, xt0, dxt0, t0, tmax, dV[j], frekvenca)   

    if j==0:
        odmiki.plot(HISTfrekUP, HISTampUP, 'c-.', label='$višanje, v=\pi/10$')
        odmiki.plot(HISTfrekDWN, HISTampDWN, 'c:', label='$padanje, v=\pi/10$')
        print('prva')    
    elif j==1:
        odmiki.plot(HISTfrekUP, HISTampUP, 'k-.', label='$višanje, v=\pi/8$')
        odmiki.plot(HISTfrekDWN, HISTampDWN, 'k:', label='$padanje, v=\pi/8$')
        print('druga')    
    elif j==2:
        odmiki.plot(HISTfrekUP, HISTampUP, 'b-.', label='$višanje, v=\pi/7$')
        odmiki.plot(HISTfrekDWN, HISTampDWN, 'b:', label='$padanje, v=\pi/7$')
        print('tretja')    
    elif j==3:
        odmiki.plot(HISTfrekUP, HISTampUP, 'g-.',  label='$višanje, v=\pi/6$')
        odmiki.plot(HISTfrekDWN, HISTampDWN, 'g:',  label='$padanje, v=\pi/6$')
        print('cetrta')    
    elif j==4:
        odmiki.plot(HISTfrekUP, HISTampUP, 'r-.', label='$višanje, v=\pi/4$')
        odmiki.plot(HISTfrekDWN, HISTampDWN, 'r:', label='$padanje, v=\pi/4$')

    odmiki.set_xlabel( '$\omega_{0}$' )
    odmiki.set_ylabel( '$x_{max}$' )
    odmiki.set_title( ' Resonančne krivulje računane s prilagodljivo dolžino koraka $\t{RK4}$')# (h='+str(korak)+')' )


odmiki.legend(loc='upper right')
odmiki.legend(loc='best')
    


#_________________________ GRAFIRANJE _________________________________________________

# grafiranje natančnost in stabilnost
#
#FIGerr= plt.figure()
#Errfig1=plt.subplot(3, 1, 1 )
#plt.plot(trk4, abs(Xt-Anxrk4), 'k', label='RK4_as')
#plt.plot(tx, abs(Trx-TAnx), 'r', label='trapezna')
#plt.plot(Itx, abs(TrIx-TAnx), 'g', label='trapezna_imp')
#plt.plot(t, abs(sol[:, 0]-Anx), 'b', label='odeint')
#plt.legend(loc='best')
#plt.xlabel( '$t$' )
#plt.ylabel( 'Err:x(t)' )
#Errfig1.set_yscale('log')
#plt.title( 'Natančnost metod pri računanju odmikov x(t)')# (h='+str(korak)+')' )
#
#Errfig2=plt.subplot(3, 1, 2 )
#plt.plot(trk4, abs(dXt-Andxrk4), 'k', label='RK4_as')
#plt.plot(tdx, abs(Trdx-TAndx), 'r', label='trapezna')
#plt.plot(Itdx, abs(TrIdx-TAndx), 'g', label='trapezna_imp')
#plt.plot(t, abs(sol[:, 1]-Andx), 'b', label='odeint')
#plt.legend(loc='best')
#plt.xlabel( '$t$' )
#plt.ylabel( 'Err:v(analitičnat)' )
#Errfig2.set_yscale('log')
#plt.title( 'Natančnost metod pri računanju hitrosti v(t)')# (h='+str(korak)+')' )
#
#Errfig3=plt.subplot(3, 1, 3 )
#plt.plot(trk4, energijaRK4, 'k', label='RK4_as')
#plt.plot(tx, E, 'r', label='trapezna')
#plt.plot(Itx, IE, 'g', label='trapezna_imp')
#plt.plot(t, Ean, 'b', label='odeint')
#plt.legend(loc='best')
#plt.ylabel('E(x(t),v(t))')
#plt.xlabel('t')
#plt.grid()
#plt.title( 'Energija nihala tekom iteracije pri računanju z metodo $\t{RK4}$')# (h='+str(korak)+')' ) 
#plt.show()

## grafiranje RK4 adaptive
#
#FIGrk4= plt.figure()
#rk4fig1=plt.subplot(3, 1, 1 )
#plt.plot(trk4, Xt, 'b', label='x(t)')
#plt.plot(trk4, dXt, 'g', label='v(t)')
#plt.legend(loc='best')
#plt.xlabel( '$t$' )
#plt.ylabel( 'amplituda' )
#plt.title( 'Rezultati reševanja enačbe matematičnega nihala s prilagodljivo dolžino koraka $\t{RK4}$')# (h='+str(korak)+')' )
#
#rk4fig3=plt.subplot(3, 1, 3 )
#plt.plot(trk4,dH, 'k', label='dolžina koraka')
#plt.legend(loc='best')
#plt.xlabel( 'korak' )
#plt.ylabel( 'dolžina koraka' )
#plt.title( 'Spreminjanje dolžine koraka tekom iteracije pri računanju z metodo $\t{RK4}$')# (h='+str(korak)+')' ) 
#
#scpyfig2=plt.subplot(3, 1, 2 )
#plt.plot(trk4, energijaRK4, 'r', label='E(x(t),v(t))')
#plt.legend(loc='best')
#plt.ylabel('amplituda')
#plt.xlabel('t')
#plt.grid()
#plt.title( 'Energija nihala tekom iteracije pri računanju z metodo $\t{RK4}$')# (h='+str(korak)+')' ) 
#plt.show()

#
## grafiranje SCIPY

#FIGscpy= plt.figure()
#scpyfig1=plt.subplot(2, 1, 1 )
#plt.plot(t, sol[:, 0], 'b', label='x(t)')
#plt.plot(t, sol[:, 1], 'g', label='v(t)')
#plt.legend(loc='best')
#plt.ylabel('amplituda')
#plt.xlabel('t')
#plt.grid()
#plt.title( 'Rezultati reševanja enačbe matematičnega nihala s prilagodljivo dolžino koraka ($\t{scipy.odeint}$)')# (h='+str(korak)+')' )
#plt.show()
#
#scpyfig2=plt.subplot(2, 1, 2 )
#plt.plot(t, Esci, 'r', label='E(x(t),v(t))')
#plt.legend(loc='best')
#plt.ylabel('amplituda')
#plt.xlabel('t')
##scpyfig2.set_yscale('log')
#plt.grid()
#plt.title( 'Energija nihala tekom iteracije pri računanju z metodo $\t{scipy.odeint}$')# (h='+str(korak)+')' ) 
#plt.show()
#
## grafiranje trapezna
# 
#FIGtr= plt.figure()
#trfig1=plt.subplot(2, 1, 1 )
#plt.plot(tx, Trx, 'b', label='x(t)')
#plt.plot(tdx, Trdx, 'g', label='v(t)')
#plt.legend(loc='best')
#plt.ylabel('amplituda')
#plt.xlabel('t')
#plt.grid()
#plt.title( 'Rezultati reševanja enačbe matematičnega nihala s trapezno metodo' )
#plt.show()
##
#trfig2=plt.subplot(2, 1, 2 )
#plt.plot(tx, E, 'r', label='E(x(t),v(t))')
#plt.legend(loc='best')
#plt.ylabel('amplituda')
#plt.xlabel('t')
##trfig2.set_yscale('log')
#plt.grid()
#plt.title( 'Energija nihala tekom iteracije pri računanju s trapezno metodo')# (h='+str(korak)+')' ) 
#plt.show() 
# 
## grafiranje implicitna trapezna
# 
#FIGtrI= plt.figure()
#trIfig1=plt.subplot(2, 1, 1 )
#plt.plot(Itx, TrIx, 'b', label='x(t)')
#plt.plot(Itdx, TrIdx, 'g', label='v(t)')
#plt.legend(loc='best')
#plt.ylabel('amplituda')
#plt.xlabel('t')
#plt.grid()
#plt.title( 'Rezultati reševanja enačbe matematičnega nihala z implicitno trapezno metodo' )
#plt.show()
##
#trIfig2=plt.subplot(2, 1, 2 )
#plt.plot(Itx, IE, 'r', label='E(x(t),v(t))')
#plt.legend(loc='best')
#plt.ylabel('amplituda')
#plt.xlabel('t')
##trIfig2.set_yscale('log')
#plt.grid()
#plt.title( 'Energija nihala tekom iteracije pri računanju  z implicitno trapezno metodo')# (h='+str(korak)+')' ) 
#plt.show()
#
## grafiranje analitična
#
#FIGAn= plt.figure()
#Anfig1=plt.subplot(2, 1, 1 )
#plt.plot(t, Anx, 'k', label='analitična')
#plt.plot(trk4, Xt, 'm:', label='RK4_as')
#plt.plot(tx, Trx, 'r:', label='trapezna')
#plt.plot(Itx, TrIx, 'g:', label='trapezna_imp')
#plt.plot(t, sol[:, 0], 'b:', label='odeint')
#plt.legend(loc='best')
#plt.ylabel('x(t)')
#plt.xlabel('t')
#plt.grid()
#plt.title( 'Rezultati x(t) enačbe matematičnega')# (h='+str(korak)+')' )
#plt.show()
#
#Anfig2=plt.subplot(2, 1, 2 )
#plt.plot(t, Andx, 'k', label='analitična')
#plt.plot(trk4, dXt, 'm:', label='RK4_as')
#plt.plot(tdx, Trdx, 'r:', label='trapezna')
#plt.plot(Itdx, TrIdx, 'g:', label='trapezna_imp')
#plt.plot(t, sol[:, 1], 'b:', label='odeint')
#plt.legend(loc='best')
#plt.ylabel('v(t)')
#plt.xlabel('t')
##Anfig2.set_yscale('log')
#plt.grid()
#plt.title( 'Rezultati v(t) enačbe matematičnega')# (h='+str(korak)+')' ) 
#plt.show()