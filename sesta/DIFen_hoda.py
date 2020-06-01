# -*- coding: utf-8 -*-

import pylab 
import numpy
#from scipy.integrate import ode


#pi=3.1415926535897932384626433

#------------------------toplotna_prevodnost-----------------------------------

#x0=21. # -15.  začetna temperatura
#xzun=-5.
#k=0.1  #Izberi metodo (in korak) za izračun dručine rešitev pri različnih vrednostih parametra

#a=0.
#b=50.
#n = 10*5.
#korak=(b-a)/(n-1)
#t = numpy.linspace( a, b, n )

#def f( x, xzun, k, t ):
#    return -k*(x-xzun)
#
#def F( x, x0, xzun, k, t ):     # compute true solution values in equal spaced and unequally spaced cases
#    return xzun+numpy.exp(-k*t)*(x0-xzun)

#________________________DODATNA_NALOGA_______________________________________
#
#x0=21. # - 15.  začetna temperatura
#xzun=-5.
#k=0.1
#A=1. # Začni z A= 1, kasneje spreminjaj tudi to vrednost. 
#
#a=0
#b=100
#n = 10*5
#korak=(b-a)/n
#t = numpy.linspace( a, b, n )#
def f( x, xzun, k, t ):
    pi=3.1415926535897932384626433
    return -k*(x-xzun)+A*numpy.sin(2.*pi*(t-10.)/24.)

#_________________________METODE_______________________________________

def euler( x, x0, xzun, k, t ): 
    """Euler's method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = euler(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:, x0, xzun, k, t ) # h=0.1
    x_METODA = rk4( x, x0, xzu
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """
    n = len( t )
    x = numpy.array( [x0] * n )
    
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        x[i+1] = x[i] + h * f( x[i],  xzun, k, t[i] )

    return x
    
def eulerSIM( x, x0, xzun, k, t ): 

    n = len( t )
    x = numpy.array( [x0] * n )
    
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        x[i+1] = x[i-1]  + 2* h * f( x[i], xzun, k, t[i] )

    return x   
    
def eulerIZB( x, x0, xzun, k, t ): 

    n = len( t )
    x = numpy.array( [x0] * n )
    
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        x[i+1] = x[i]  +  h * f( x[i]+h*f(x[i], xzun, k, t[i])/2 , xzun, k, t[i] )
        
    return x   
    
    
def eulerTRAPEZ( x, x0, xzun, k, t ): 
  
    n = len( t )
    x = numpy.array( [x0] * n )
    
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
#        x[i+1] = x[i]  +  h * f( x[i]+h*f(x[i], xzun, k, t[i])/2 , xzun, k, t[i] )
        x[i+1] = x[i] + h * f( x[i],  xzun, k, t[i] )
        eps = 10.**(-10)
        Y=x[i+1]+eps
        while abs(x[i+1]-Y)>eps:
            Y=x[i+1]
            x[i+1] = x[i]  +  h * (f( x[i], xzun, k, t[i] )+f( x[i+1], xzun, k, t[i+1] ))/2
    return x   
    
    
    
def rk4( x0, xzun, k, t ): 
    """Fourth-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rk4(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len( t )
    x = numpy.array( [ x0 ] * n )
    
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( x[i], xzun, k, t[i] )
        k2 = h * f( x[i] + 0.5 * k1, xzun, k, t[i] + 0.5 * h )
        k3 = h * f( x[i] + 0.5 * k2, xzun, k, t[i] + 0.5 * h )
        k4 = h * f( x[i] + k3, xzun, k, t[i+1] )
        x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0

    return x


def ADrk4( x0, xzun, k, t ): 
    
    eps = 10.**(-2) # desired accuracy
    n = len( t ) # length of the interval to calculate values x on
    x = numpy.array( [ x0 ] * n ) # array for saving calculated values info
    HH = numpy.array( [ 0.001 ] * n ) # array for saving step size info
    
    for i in range( n - 1 ):
        
        h = t[i+1] - t[i] # oroginal step length 
        HH[i]=h # write first used step length to array
        
        k1 = h * f( x[i], xzun, k, t[i] )
        k2 = h * f( x[i] + 0.5 * k1, xzun, k, t[i] + 0.5 * h )
        k3 = h * f( x[i] + 0.5 * k2, xzun, k, t[i] + 0.5 * h )
        k4 = h * f( x[i] + k3, xzun, k, t[i+1] )
        x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6. # calculate value witk RK4 method
        
        l=10
        XX=x[i+1] # calculate value witk RK4 method
        XXX=x[i+1]+10.*eps # entering the while loop condition

        while abs(XXX-XX)>eps and l<100000: # compare two latest different step calculations for desired accuracy
            
            H = h/l # dividing calculation step H by intiger l
            XXX=XX # rewrite entering condition to latest calculation of x[i+1] 
            T=t[i] # setting start step point
            X=x[i] # setting start step value
            
            for j in range( 0, l-1 ): # caluculating with 'half' steps; l times
                
                k1 = h * f( X, xzun, k, (T+H*j) )
                k2 = h * f( X + 0.5 * k1, xzun, k, (T+H*j) + 0.5 * H )
                k3 = h * f( X + 0.5 * k2, xzun, k, (T+H*j) + 0.5 * H )
                k4 = h * f( X + k3, xzun, k, (T+H*(j+1)))
                XX = X + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0 # XX rewritten to more accurate value x[i+1]
#                print(j)
                j=j+1 #increase for next step in for loop
                
#            print(l)
#            print(XX)
                
            l=l*10  # dubble l for further division of step if necessary
            
            x[i+1] = XX #write last most accurate calculation of  x[i+1]
            HH[i]=H # write last used step length
        
    return x, HH

#__________________________________RISANJE_GRAFOV__________________________________________________    


## compute various numerical solutions
#x_euler = euler( x, x0, xzun, k, t )
#x_eulerSIM = eulerSIM( x, x0, xzun, k, t )
#x_eulerIZB = eulerIZB( x, x0, xzun, k, t ) 
#x_eulerTRAPEZ = eulerTRAPEZ( x, x0, xzun, k, t )
#x_rk4 = rk4( x, x0, xzun, k, t )
##t_rkf, x_rkf = rkf( f, a, b, x0, 1e-6, 1.0, 0.01 ) # unequally spaced t
#
##x_py=scipy.integrate.ode(f( x, xzun, k, t ))
#
## compute true solution values in equal spaced and unequally spaced cases
#x_PRAVI = F( x, x0, xzun, k, t )
##xrkf = sin(t_rkf) - t_rkf*cos(t_rkf)
#
#
#FIG=figure()
#fig1=subplot(2, 1, 1 )
#plot( t, x_PRAVI, 'k-', t, x_euler, 'r--', t, x_eulerSIM, 'g:', t, x_eulerIZB, 'b:', t, x_eulerTRAPEZ, 'y-' , t, x_rk4, 'm-.')#, t, x_py, 'c-.')
#xlabel( '$t$' )
#ylabel( '$T$ [C]' )
#title( 'Rezultati T(t) reševanja toplotne enačbe z različnimi metodami (h='+str(korak)+')' )
#legend( ( 'An', 'E', 'ES', 'EI', 'ET', 'RK4' ), loc='upper right' )
#
#fig2=subplot( 2, 1, 2 )
#plot( t, abs(x_euler-x_PRAVI), 'r-', t, abs(x_eulerSIM-x_PRAVI), 'g-', t, abs(x_eulerIZB-x_PRAVI), 'b-', t, abs(x_eulerTRAPEZ-x_PRAVI), 'y-' , t, abs(x_rk4-x_PRAVI), 'm-')# t, abs(x_py-x_PRAVI), 'c-'  )
#xlabel( '$t$' )
#ylabel( '$|T_{Nm} - T_{An}|$  [^{\circ}C]' )
#title( 'Odstopanja numeričnih rezultatov od analitične rešitve (h='+str(korak)+')')
#legend( ( 'E', 'ES', 'EI', 'ET', 'RK4' ), loc='upper right' )
#fig2.set_yscale('log')

#fig2=subplot( 2, 1, 2 )
#plot( t, abs(x_euler-x_PRAVI), 'r--', t, abs(x_eulerIZB-x_PRAVI), 'b:', t, abs(x_eulerTRAPEZ-x_PRAVI), 'y-' , t, abs(x_rk4-x_PRAVI), 'm-.'  )
#xlabel( '$t$' )
#ylabel( '$T_{Nm} - T_{An}$ [C]' )
#title( 'Odstopanja numeričnih rezultatov od analitične rešitve (h='+str(korak)+')')
#legend( ( 'E', 'EI', 'ET', 'RK4' ), loc='upper right' )
#fig2.set_yscale('log')

#__________________________________RISANJE_GRAFOV__________________________________________________    

crta=['k-','k--','k-.','k:','r--','g--','b--.','r-']
FIG=pylab.figure()
i=1
k=0.1 # 0.1 0.4 0.8 1.5 3
A=1.
while i <1.5:
    
#    A=A+0.2*i
    x0=21. # 21.  začetna temperatura
    xzun=-5.
#        k=0.4 #Izberi metodo (in korak) za izračun družine rešitev pri različnih vrednostih parametra
    
    a=0
    b=100
    n = 10**(i)*5 # uzračun dolžine koraka 
#    n = 10**(4)*5  # Izberi dolžino koraka 'h=0.001'
    t = numpy.linspace( a, b, n )

#    
#    x_METODA = euler( x, x0, xzun, k, t ) # h=0.001
#    x_METODA = eulerSIM( x, x0, xzun, k, t ) # h=0.001
#    x_METODA = eulerIZB( x, x0, xzun, k, t ) # h=0.0001
#    x_METODA = eulerTRAPEZ( x, x0, xzun, k, t ) # h=0.1
#    x_METODA = rk4( x0, xzun, k, t ) # h=0.001
    x_METODA,koraki = ADrk4( x0, xzun, k, t ) # rk4 in prilagodljiv korak

#    x_PRAVI = F( x, x0, xzun, k, t )    
#    x_PRAVI = ADrk4( x, x0, xzun, k, t ) # rk4 in prilagodljiv korak
    
    fig1=pylab.subplot(2, 1, 1 )
    if x0==21.:
        pylab.ylim([-16, x0+5]) 
    else:
        pylab.ylim([x0-5, 0]) 
        
    pylab.plot( t, x_METODA, crta[i]) #, t, x_euler, 'r--', t, x_eulerSIM, 'g:', t, x_eulerIZB, 'b:', t, x_eulerTRAPEZ, 'y-' , t, x_rk4, 'm-.')#, t, x_py, 'c-.')
#    plot( t, x_PRAVI, crta[i+1]) 
    pylab.xlabel( '$t$' )
    pylab.ylabel( '$T$  [$^{\circ}$C]' )
#    title( 'Rezultati reševanja toplotne enačbe pri $k='+str(k)+'$ za različne dolžine koraka (metoda RK4)')# (h='+str(korak)+')' ) '''(E)(ES)(EI)(ET)(RK4)'''
#    title( 'Rezultati reševanja večparametrične toplotne enačbe za različne dolžine koraka (metoda RK4)')# (h='+str(korak)+')' ) '''(E)(ES)(EI)(ET)(RK4)'''
#    title( 'Rezultati reševanja večparametrične toplotne enačbe za korak h=0.001 in različne A (metoda RK4)')# (h='+str(korak)+')' ) '''(E)(ES)(EI)(ET)(RK4)'''

#    legend( ( 'An', 'E', 'ES', 'EI', 'ET', 'RK4' ), loc='upper right' )
#    legend(  ('h=10',  'h=1',  'h=0.1',  'h=0.01',  'h=0.001' , 'h=0.0001', 'h=0.00001'), loc='upper right' )
#    legend(  ('h=10', 'h=1',  'h=0.1',  'h=0.01',  'h=0.001' , 'h=0.0001'), loc='upper right' )

#    legend(  ('h=1',  'h=0.1',  'h=0.01',  'h=0.001' , 'h=0.0001'), loc='lower right' )
#    legend( ( 'A=0.1', 'A=0.3', 'A=0.7', 'A=1.3', 'A=2.1','A=3.1'), loc='upper right' )
#
    fig2=pylab.subplot( 2, 1, 2 )
    pylab.plot( t, koraki, crta[i-1])
#    plot( t, abs(x_METODA-x_PRAVI), crta[i]) #, t, abs(x_eulerSIM-x_PRAVI), 'g-', t, abs(x_eulerIZB-x_PRAVI), 'b-', t, abs(x_eulerTRAPEZ-x_PRAVI), 'y-' , t, abs(x_rk4-x_PRAVI), 'm-')# t, abs(x_py-x_PRAVI), 'c-'  )
#    xlabel( '$t$' )
#    ylabel( '$|T_{Nm} - T_{An}|$  [$^{\circ}$C]' )
#    title( 'Odstopanja numeričnih rezultatov od analitične rešitve')# (h='+str(korak)+')')
##    legend( ( 'E', 'ES', 'EI', 'ET', 'RK4' ), loc='upper right' )
    fig2.set_yscale('log')
    pylab.ylim([10e-16, 10]) 
#
    i=i+1

#
#fig1.legend(  'h=10',  'h=1',  'h=0.1',  'h=0.01',  'h=0.001' , 'h=0.0001', loc='upper right' )
#fig2.legend( 'h='+str(korak[0]),  'h='+str(korak[1]),  'h='+str(korak[2]),  'h='+str(korak[3]),  'h='+str(korak[4]), 'h='+str(korak[5]), loc='upper right' )




