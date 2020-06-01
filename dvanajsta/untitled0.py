# -*- coding: utf-8 -*-

import scipy as sc
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import special
from pylab import *
import time

#******************************************************
#******************************************************

def b(m_max,n_max):
    #izračuna vektor b_j=b_m'n'
    b=[0.0 for j in range((m_max+1)*n_max)]
    for m_ in range(0,m_max+1,1):
        for n_ in range(1,n_max+1,1):
            k1=-2.0/(2*m_+1)
            b[m_*n_max+n_-1]=k1*sc.special.beta(2*m_+3,n_+1)
            #print('m_=%f, n_=%f, b=%f' %(m_,n_,b[m_*n_max+n_-1]))

    return b


def A(m_max, n_max):
    #najprej ustavrimo matriko A in ji določimo dimenzije
    D_matrix=(m_max+1)*(n_max)   #=dimenzija matrike
    A=[[0.0 for j in range(D_matrix)] for k in range(D_matrix)]
    for m in range(0,m_max+1,1):
        for m_ in range(0,m_max+1,1):
            if m==m_:
                #sedaj smo znotraj enega bloka
                for n in range(1,n_max+1,1):
                    for n_ in range(1,n_max+1,1):
                        #sedaj smo v eni celici:
                        k1=-np.pi/2*n*n_*(3+4*m)/(2+4*m+n+n_)
                        A[n_max*m_+n_-1][n_max*m+n-1]=k1*sc.special.beta(n+n_-1,3+4*m)
                        #print('m=%f, m_=%f, n=%f, n_=%f, A=%f' %(m,m_,n,n_, A[n_max*m_+n_-1][n_max*m+n-1]))
    return A

def C(m_max, n_max):
    k1=np.dot(np.linalg.inv(A(m_max, n_max)),b(m_max, n_max))
    return -32.0/(np.pi)*np.dot(b(m_max,n_max),k1)

#************************************************************************
t1=time.clock()
print(C(6,1))
print(C(6,2))
print(C(3,1))
print(C(3,2))


t2=time.clock()
print('\n Cas: %f' %(t2-t1))

#*****************************************************************
#************************    RIŠEMO   ****************************
