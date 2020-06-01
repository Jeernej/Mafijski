# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:29:15 2016

@author: jernej
"""

import numpy
#----------equation-and-constants--------------------


def flow( x, t ):  # temperature curve function
     pi=3.1415926535897932384626433
     A=1
     xzun=-5. 
     k=0.1
     
     f=-k*(x-xzun)+A*numpy.sin(2.*pi*(t-10.)/24.)
     
     return float(f )

#-----------------algorithm-----------------------

#a=0 # interval start    
#b=100 # interval end    
#n = 10**(2)  # initial step length to h=0.1
#t = numpy.linspace( a, b, n ) # interval points array 
#u = numpy.linspace( a, b, n-1 ) # interval points array 
#
#x0=-15. # initial condition
#
#def ADrk4(  x0, xzun, k, t ): 
#    
#    eps = 10.**(-10) # desired accuracy
#    n = len( t ) # length of the interval to calculate values x on
#    x = numpy.array( [ x0 ] * n ) # array for saving calculated values info
#    HH = numpy.array( [ x0 ] * (n-1) ) # array for saving step size info
#    
#    for i in range( n - 1 ):
#        
#        h = t[i+1] - t[i] # set to initial step length 
#        HH[i]=h # write first used step length to array
#        
#        k1 = h * f( x[i], xzun, k, t[i] )
#        k2 = h * f( x[i] + 0.5 * k1, xzun, k, t[i] + 0.5 * h )
#        k3 = h * f( x[i] + 0.5 * k2, xzun, k, t[i] + 0.5 * h )
#        k4 = h * f( x[i] + k3, xzun, k, t[i+1] )
#        x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6. # calculate value witk RK4 method
#        
#        l=1
#        XX=x[i+1] # calculate value witk RK4 method
#        XXX=x[i+1]+10.*eps # entering the while loop condition
#
#        while abs(XXX-XX)>eps: # compare two latest different step calculations for desired accuracy
#            
#            H = h/10**l # dividing calculation step H by intiger l
#            XXX=XX # rewrite entering condition to latest calculation of x[i+1] 
#            T=t[i] # setting start step point
#            X=x[i] # setting start step value
#            
#            for j in range( 0, (10**l)-1 ): # caluculating with 'half' steps; l times
#                
#                k1 = h * f( X, xzun, k, (T+H*j) ) # calculating with different step H
#                k2 = h * f( X + 0.5 * k1, xzun, k, (T+H*j) + 0.5 * H )
#                k3 = h * f( X + 0.5 * k2, xzun, k, (T+H*j) + 0.5 * H )
#                k4 = h * f( X + k3, xzun, k, (T+H*(j+1)))
#                XX = X + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0 # XX rewritten to more accurate value x[i+1]
##                print(XX)
#                j=j+1 #increase for next step in for loop
#                
##            print(H)
#            print(abs(XXX-XX))
#            if l>10000:
#                print('spam & eggs')
#                break
#            l=l+1  # dubble l for further division of step if necessary
#            
#        x[i+1] = XX #write last most accurate calculation of  x[i+1]
#        HH[i]=H # write last used step length
##        break
#    return x, HH
    
def RK4_step(x, dt):    # iz x(t) dobis x(t + dt)
    n = len(x)
    k1 = [ dt * k for k in flow(x,dt)]
    x_temp = [ x[i] + k1[i] / 2 for i in range(n) ]
    k2 = [ dt * k for k in flow(x_temp,dt) ]
    x_temp = [ x[i] + k2[i] / 2 for i in range(n) ]
    k3 = [ dt * k for k in flow(x_temp,dt) ]
    x_temp = [ x[i] + k3[i] for i in range(n) ]
    k4 = [ dt * k for k in flow(x_temp,dt) ]
    for i in range(n):
        x[i] += (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6

def RK4_adaptive_step(x, dt,  accuracy):  # from Numerical Recipes
    SAFETY = 0.9; PGROW = -0.2; PSHRINK = -0.25;
    ERRCON = 1.89E-4; TINY = 1.0E-30
    n = len(x)
    scale = flow(x,dt)
    scale = [ abs(x[i]) + abs(scale[i] * dt) + TINY for i in range(n) ]
    while True:
        dt /= 2
        
        x_half = [ x[i] for i in range(n) ]
        RK4_step(x_half, dt)
        RK4_step(x_half, dt)
        dt *= 2
        
        x_full = [ x[i] for i in range(n) ]
        RK4_step(x_full, dt)
        
        Delta = [ x_half[i] - x_full[i] for i in range(n) ]
        error = max( abs(Delta[i] / scale[i]) for i in range(n) ) / accuracy
        if error <= 1:
            break;
        dt_temp = SAFETY * dt * error**PSHRINK
        if dt >= 0:
            dt = max(dt_temp, 0.1 * dt)
        else:
            dt = min(dt_temp, 0.1 * dt)
        if abs(dt) == 0.0:
            raise OverflowError("step size underflow")
    if error > ERRCON:
        dt *= SAFETY * error**PGROW
    else:
        dt *= 5
    for i in range(n):
        x[i] = x_half[i] + Delta[i] / 15
    return dt
    
#----------calculation--------------------
a=0 # interval start    
b=100 # interval end    
n = 10**(2)  # initial step length to h=0.1
t = numpy.linspace( a, b, n ) # interval points array 
u = numpy.linspace( a, b, n-1 ) # interval points array 
x0=-15. # initial condition

x = numpy.array( [ x0 ] * n ) # array for saving calculated values info
HH = numpy.array( [ x0 ] * (n-1) ) # array for saving step size info
    
eps = 10.**(-10) # desired accuracy
#x_values,step_size = ADrk4(  x0, xzun, k, t ) 
x_values,step_size = RK4_adaptive_step(x, n,  eps)  # from Numerical Recipes

#----------ploting--------------------

fig1=subplot(2, 1, 1 )
plot( t, x_values, 'r--')
xlabel( '$t$' )
ylabel( '$T$  [$^{\circ}$C]' )

fig2=subplot(2, 1, 2 )
plot( u, step_size, 'b--') 
xlabel( '$t$' )
ylabel( 'step size' )
fig2.set_yscale('log')