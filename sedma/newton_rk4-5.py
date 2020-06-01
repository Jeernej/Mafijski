## -*- coding: utf-8 -*-

import scipy as sc
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#_____ RK4 adaptive step rešitev________________________________________________


#    Adaptive step size ODE solver with RK5 (embedded RK4)
#
#    Formulation and notation taken from Numerical Recipes in Fortran 
#    77, Second Edition (1992) by William H. Press, Brian P. Flannery, 
#    Saul A. Teukolsky, and William T. Vetterling.
#    ISBN-10: 052143064X 
#
#    Alexander Miles - Last edit: 7/30/12

#from time import time # Import time-keeping library
#tstart = time()       # Define starting time
#print ("[--.--] Starting. Importing libraries.")
#from pylab import *   # Import libraries for plotting results
#close('all')          # Close previously opened plots


## Cash-Karp Parameters - From literature

#print ("[%4.3f] Defining parameters and functions." % (time()-tstart))



def control(ddxt, tol, xt0, dxt0, t0, h, tmax ): #, v=False):
    '''
#    This funciton takes in a python function that returns the derivative,
#    a tolerance for error between RK5 and RK4, initial conditions on y (dependant) 
#    and t (independant) as well as an initial step size.
#    
#    Keyword arguments:
#    v - Verbose
    '''

#    tstart = time()
#    if v==True: print ("[%4.3f] Solving with initial condition (%0.2f, %0.2f), step size of %0.2f from t=0...%0.2f" % ((time()-tstart), y0, t0, h, tmax))

    Xt = [xt0]      # Set up the initial conditions on x(t)
    dXt = [dxt0]      # Set up the initial conditions on dx(t)
    energijaRK4 = [1-np.cos(xt0)+(dxt0)**2/2]      # Set up the initial conditions on energy

    t = [t0]      # and t while creating the output lists
    H = [h]   
   
#    t_curr, x_curr, count, ncount = t0, y0, 0, 0 # Setup counters and trackers
    t_curr = t0
    xt_curr = xt0 # Setup counters and trackers
    dxt_curr =  dxt0  # Setup counters and trackers

    while t_curr < tmax:
        
        dxt_next, h, h2 = stepper(ddxt, xt_curr, dxt_curr, h, tol, optimizacija=True) # pridobi optimalni korak  h2
        
        xt_next, h, h1 = stepper2( xt_curr, dxt_curr, h, h, tol, optimizacija=True)  # pridobi optimalni korak h1
        
#        if h1 < 0.9*h :
##            if v==True: print ("[%4.3f] Reduced step size from %0.4e to %0.4e at t = %0.2f" % ((time()-tstart),h, h1, t_curr))
#            h = h1
#        elif h1 > 1.1*h:
##            if v==True: print ("[%4.3f] Increased step size from %0.4e to %0.4e at t = %0.2f" % ((time()-tstart),h, h1, t_curr))
##            h = h1
#        print(h1)
#        print(h2)
        
        if h1 < h2: # uporabi krajši optimalni korak od obeh in računaj z njim šeenkrat obe

            h = h1
            dxt_next, h , h2  = stepper(ddxt, xt_curr, dxt_curr, h, tol, optimizacija=False)
            xt_next, h, h1 = stepper2( xt_curr, dxt_curr, h, h, tol, optimizacija=False)

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

            dxt_next, h, h2  = stepper(ddxt, xt_curr, dxt_curr, h, tol, optimizacija=False)
            xt_next, h, h1 = stepper2( xt_curr, dxt_curr, h, h, tol, optimizacija=False)

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

            dxt_next, h, h2  = stepper(ddxt, xt_curr, dxt_curr, h, tol, optimizacija=False)
            xt_next, h, h1  = stepper2( xt_curr, dxt_curr, h, h, tol, optimizacija=False)
            
            Xt.append(xt_next)
            dXt.append(dxt_next)
            energijaRK4.append(1-np.cos(xt_next)+(dxt_next)**2/2)
            t.append(t_curr+h)
            H.append(h)
            xt_curr = xt_next
            dxt_curr= dxt_next
            t_curr = t_curr+h
            
#            print('-'+str(h1))
#            print('-'+ str(h2))            
            
            
#        print(t_curr)
            
#            ncount += 1
#        count += 1        
#    if v==True: print ("[%4.3f] Done. %i iterations, %i points" % ( (time()-tstart), count, ncount ))
    return Xt, dXt, energijaRK4, t ,H
   
   
a2,   a3,  a4,  a5,  a6      =        1/5.,    3/10.,       3/5.,            1.,        7/8.
b21, b31, b32, b41, b42, b43 =        1/5.,    3/40.,      9/40.,         3/10.,      -9/10., 6/5.
b51, b52, b53, b54           =     -11/54.,     5/2.,    -70/27.,        35/27.
b61, b62, b63, b64, b65      = 1631/55296., 175/512., 575/13824., 44275/110592.,   253/4096. 
c1,   c2,  c3,  c4,  c5, c6  =     37/378.,       0.,   250/621.,      125/594.,          0.,  512/1771.
c1star, c2star, c3star, c4star, c5star, c6star = 2825/27648., 0.,  18575/48384.,13525/55296., 277/14336., 1/4.
    
def stepper(deriv, t, y, h, tol, optimizacija):
    '''
#   This function is called by the control function to take a single step forward. The inputs are the derivative function,
#   the previous time and function value, the step size in time (h), and the tolerance
#   for error between 5th order Runge-Kutta and 4th order Runge-Kutta.
    '''


    k1 = h*deriv(t,y)
    k2 = h*deriv(t+a2*h,y+b21*k1)
    k3 = h*deriv(t+a3*h,y+b31*k1+b32*k2)
    k4 = h*deriv(t+a4*h,y+b41*k1+b42*k2+b43*k3)
    k5 = h*deriv(t+a5*h,y+b51*k1+b52*k2+b53*k3+b54*k4)
    k6 = h*deriv(t+a6*h,y+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5)  # za izračun z natančnejšo metodo (reda 5) za izbiro koraka
   
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
            
            k1 = h*deriv(t,y)
            k2 = h*deriv(t+a2*h,y+b21*k1)
            k3 = h*deriv(t+a3*h,y+b31*k1+b32*k2)
            k4 = h*deriv(t+a4*h,y+b41*k1+b42*k2+b43*k3)
            k5 = h*deriv(t+a5*h,y+b51*k1+b52*k2+b53*k3+b54*k4)
            k6 = h*deriv(t+a6*h,y+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5)
            y_n_plus_1      = y +     c1*k1 +     c2*k2 +     c3*k3 +     c4*k4 +     c5*k5 +     c6*k6
#            print(h1)
    else:
        pass
#    h1 = h                       
#    print(h1)
    return y_n_plus_1, h, h1

def stepper2( xt_curr, dxt_curr, h, H, tol, optimizacija):
    '''
#   This function is called by the control function to take a single step forward. The inputs are the derivative function,
#   the previous time and function value, the step size in time (h), and the tolerance
#   for error between 5th order Runge-Kutta and 4th order Runge-Kutta.
    '''
    
    k1 = h*dxt_curr
    k2 = h*dxt( ddxt, xt_curr+a2*h, dxt_curr+b21*k1, H, tol )
    k3 = h*dxt( ddxt, xt_curr+a3*h, dxt_curr+b31*k1+b32*k2, H, tol )
    k4 = h*dxt( ddxt, xt_curr+a4*h, dxt_curr+b41*k1+b42*k2+b43*k3, H, tol )
    k5 = h*dxt( ddxt, xt_curr+a5*h, dxt_curr+b51*k1+b52*k2+b53*k3+b54*k4, H, tol )
    k6 = h*dxt( ddxt, xt_curr+a6*h, dxt_curr+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5, H, tol )# za izračun z natančnejšo metodo (reda 5) za izbiro koraka
    
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
            k2 = h*dxt( ddxt, xt_curr+a2*h, dxt_curr+b21*k1, H, tol )
            k3 = h*dxt( ddxt, xt_curr+a3*h, dxt_curr+b31*k1+b32*k2, H, tol )
            k4 = h*dxt( ddxt, xt_curr+a4*h, dxt_curr+b41*k1+b42*k2+b43*k3, H, tol )
            k5 = h*dxt( ddxt, xt_curr+a5*h, dxt_curr+b51*k1+b52*k2+b53*k3+b54*k4, H, tol )
            k6 = h*dxt( ddxt, xt_curr+a6*h, dxt_curr+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5, H, tol )       
            y_n_plus_1      = xt_curr +     c1*k1 +     c2*k2 +     c3*k3 +     c4*k4 +     c5*k5 +     c6*k6
#            print(h1)
    else:
        pass
        
    return y_n_plus_1, h, h1
    
#### From here on down is a test case. Comment out for actual use.

#def dydt(t,y):
#    gamma = 2.0
#    return -gamma*y

def ddxt( xt, dxt ):
    b = 0
    c = 1
    ddxt=-b*dxt - c*np.sin(xt)
    return ddxt 

def dxt( ddxt, xt_curr, dxt_curr, H, tol ):
    dxt_next, h, h2 = stepper(ddxt, xt_curr, dxt_curr, H, tol, optimizacija=True)
#    dxt=y    
    return dxt_next

pi=3.1415926535897932384626433

t0=0.
xt0  = pi/2.             # Initial conditions
dxt0 = 0.           #
h    = 0.01           # Initial step size
tmax = 30.*pi         # End of time interval
#npts = int(tmax/h)  #  Number of points
tol  = 10**(-7.)     # The desired error between 4th and 5th order
                     # note that this IS NOT the error between numeric
                     # solution and the actual solution

Xt, dXt, energijaRK4, koraki ,dH = control( ddxt, tol, xt0, dxt0, t0, h, tmax) #, v=True)


#show()
FIGrk4= plt.figure()
rk4fig1=plt.subplot(3, 1, 1 )
plt.plot(koraki, Xt, 'b', label='x(t)')
plt.plot(koraki, dXt, 'g', label='v(t)')
plt.legend(loc='best')
plt.xlabel( '$t$' )
plt.ylabel( 'amplituda' )
plt.title( 'Rezultati reševanja enačbe matematičnega nihala s prilagodljivo dolžino koraka $\t{RK4}$')# (h='+str(korak)+')' )

rk4fig2=plt.subplot(3, 1, 3 )
plt.plot(koraki,dH, 'k', label='dolžina koraka')
plt.legend(loc='best')
plt.xlabel( 'korak' )
plt.ylabel( 'dolžina koraka' )
plt.title( 'Spreminjanje dolžine koraka tekom iteracije pri računanju z metodo $\t{RK4}$')# (h='+str(korak)+')' ) 

scpyfig2=plt.subplot(3, 1, 2 )
plt.plot(koraki, energijaRK4, 'r', label='E(x(t),v(t))')
plt.legend(loc='best')
plt.ylabel('amplituda')
plt.xlabel('t')
plt.grid()
plt.title( 'Energija nihala tekom iteracije pri računanju z metodo $\t{RK4}$')# (h='+str(korak)+')' ) 
plt.show()

#____SCIPY rešitev__________________________________________________________


#def ddx( t, y ):
#    return x''(t) = - b*x'(t) - c*sin(x(t)) 

#def dx( t, y ):
#    return x'(t) = y(t)

# Let y be the vector [theta, omega]. We implement this system in python as:

pi=3.1415926535897932384626433
t = np.linspace(0, 30*pi , 30*pi*100)

def nihalo(y, t):
    
    b = 0
    c = 1
    xt, dxt = y
    
    ddxt = [dxt, -b*dxt - c*np.sin(xt)]
    return ddxt

#y=[xt,dxt]

pribl = [xt0, dxt0] # začetna pogoja  [xt0,dxt0]

sol = odeint(nihalo, pribl, t) # začetna 

energija=1-np.cos(sol[:, 0])+(sol[:, 1])**2/2

# grafiranje

FIGscpy= plt.figure()
scpyfig1=plt.subplot(2, 1, 1 )
plt.plot(t, sol[:, 0], 'b', label='x(t)')
plt.plot(t, sol[:, 1], 'g', label='v(t)')
plt.legend(loc='best')
plt.ylabel('amplituda')
plt.xlabel('t')
plt.grid()
plt.title( 'Rezultati reševanja enačbe matematičnega nihala s prilagodljivo dolžino koraka ($\t{scipy.odeint}$)')# (h='+str(korak)+')' )
plt.show()
#
scpyfig2=plt.subplot(2, 1, 2 )
plt.plot(t, energija, 'r', label='E(x(t),v(t))')
plt.legend(loc='best')
plt.ylabel('amplituda')
plt.xlabel('t')
#scpyfig2.set_yscale('log')
plt.grid()
plt.title( 'Energija nihala tekom iteracije pri računanju z metodo $\t{scipy.odeint}$')# (h='+str(korak)+')' ) 
plt.show()


#_____ trapezna metoda________________________________________________






#_____ analitična rešitev________________________________________________

#
#def Analiticna(maxT,dt):
#    x0=1
#    w0=1
#    cas=int(round(maxT*1.0/dt))
#    c0=sc.sin(x0*0.5)
#    c1=sc.special.ellipk(c0*c0)
#    A=[2*np.arcsin(c0*sc.special.ellipj(c1-j*maxT*1.0*w0/cas,c0*c0)[0]) for j in range(cas)]
#    return A


def Anlt(t,xt0,dxt0):
    
    w0=1.
    Anx=[xt0]
    Andx=[dxt0]
    E0=1-np.cos(xt0)+(dxt0)**2/2
    E=[E0]
    
    c0=sc.sin(xt0*0.5)
    K=sc.special.ellipk(c0*c0)
    
    j=0
    while j < len(t)-1:
        
        sn,cn,dn,ph =sc.special.ellipj(K-w0+t[j], c0**2.)

        anx=2*np.arcsin(c0*cn) # odmik
        Anx.append(anx)
        
#        andx=-2* c0* w0 *dn* sn/(1-(cn*c0)**2.) # odvod odmika 
        if Anx[j-1] > Anx[j]:
            andx=-np.sqrt(2*(np.cos(anx)-np.cos(xt0))* w0**2)
        else:
            andx=np.sqrt(2*(np.cos(anx)-np.cos(xt0))* w0**2)
        
        Andx.append(andx)
        
        energija=1-np.cos(anx)+((andx)**2)/2
        E.append(energija)
        
        j=j+1
        
    return Anx, Andx, E
    
t = np.linspace(0, 30*pi , 30*pi*100)
Anx, Andx, E =  Anlt(t,xt0,dxt0)
 
FIGAn= plt.figure()
Anfig1=plt.subplot(2, 1, 1 )
plt.plot(t, Anx, 'b', label='x(t)')
plt.plot(t, Andx, 'g', label='v(t)')
plt.legend(loc='best')
plt.ylabel('amplituda')
plt.xlabel('t')
plt.grid()
plt.title( 'Rezultati analitične rešitve enačbe matematičnega')# (h='+str(korak)+')' )
plt.show()

Anfig2=plt.subplot(2, 1, 2 )
plt.plot(t, E, 'r', label='E(x(t),v(t))')
plt.legend(loc='best')
plt.ylabel('amplituda')
plt.xlabel('t')
#scpyfig2.set_yscale('log')
plt.grid()
plt.title( 'Energija nihala tekom iteracije računana analitično')# (h='+str(korak)+')' ) 
plt.show()