# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:21:17 2021

@author: THINKPAD
"""
import numpy as np
from sympy.physics.wigner import gaunt
import scipy as sp
from sympy import *
from scipy import special
from sympy.physics.wigner import clebsch_gordan

import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution as DE

import math
import numpy.matlib 

#%%

class AtomGaussian(): # give a system which incoulde the initial condition
    
    def __init__(self):
        self.centre= np.array([0.0, 0.0, 0.0])
        self.alpha = 0.0  #Gaussion paprameter
        self.volume = 0.0
        self.weight = 2.828427150
        self.n = 0 # number of parents gaussians for this guassian function, 
                        # used for recording overlap gaussian information
        self.sh_overlap = 0
        self.direc = np.array([0.0, 0.0,0.0])
        self.l = 1 # orbital angular momentum
        self.m = 0 # magnetic quantum number
        self.n_qm = 1 #!!! principle quantum number = 0 for now
        
        
def atomIntersection(a = AtomGaussian(),b = AtomGaussian()):
    
    c = AtomGaussian()
    c.alpha = a.alpha + b.alpha

    #centre 
    c.centre = (a.alpha * a.centre + b.alpha * b.centre)/c.alpha; 
    
    #c.sh_overlap = SHoverlap(a,b)
    
    #intersection_volume
    d = a.centre - b.centre
    d_sqr = d.dot(d)  #The distance squared between two gaussians
     
    c.weight = a.weight * b.weight * np.exp(- a.alpha * b.alpha/c.alpha * d_sqr)  
    
    c.volume = (c.weight * (np.pi/c.alpha) ** 1.5) 
    #if a.n and b.n ==1:
         
    # Set the numer of atom of the overlap gaussian
    c.n = a.n + b.n
    
    #ratio = c.volume/ ((np.pi/a.alpha)**1.5 * a.weight)
    
    return  c


def Normalize(alpha,l):
    Norm = np.sqrt(2*(2*alpha)**(l + 3/2)/special.gamma(l+ 3/2))
    return Norm


def SHoverlap(a = AtomGaussian(), b = AtomGaussian()):
   
    c = AtomGaussian()
    c.alpha = a.alpha + b.alpha
    c.centre = (a.alpha * a.centre + b.alpha * b.centre)/c.alpha
    
    direc_1 = a.direc
    direc_2 = b.direc
    
    com_direc = np.array([direc_1[0]*direc_2[0] , direc_1[1]*direc_2[1], direc_1[2]*direc_2[2]])
    co_mid = np.linalg.norm(com_direc)
    com_direc_norm = np.array([com_direc[0]/co_mid, com_direc[1]/co_mid, com_direc[2]/co_mid])
    
    #print(com_direc_norm)
    z_axis = np.array([0,0,1])

    # T_location 是目标向量
    T_location_norm = com_direc_norm
    
	# originVector是原始向量
    originVector = z_axis
    
    
    
    if (T_location_norm == originVector).all():
        R = b.centre - a.centre
        
        
    else:
        #print(T_location_norm)
        # @是向量点乘
        sita = math.acos(T_location_norm@originVector)
        n_vector_1 = np.cross(T_location_norm ,originVector )
        n_vec_norm = np.linalg.norm(n_vector_1)
        n_vector = np.array([n_vector_1[0]/n_vec_norm , n_vector_1[1]/n_vec_norm, n_vector_1[2]/n_vec_norm])
        
        #n_vector_invert = Matrix((
        #[0,-n_vector[2],n_vector[1]],
        #[n_vector[2],0,-n_vector[0]],
        #[-n_vector[1],n_vector[0],0]
        #))
        n_vector_invert = np.ndarray([3,3])
        n_vector_invert[0,0] = n_vector_invert[1,1] = n_vector_invert[2,2] = 0
        n_vector_invert[0,1] = -n_vector[2]
        n_vector_invert[0,2] = n_vector[1]
        n_vector_invert[1,0] = n_vector[2]
        n_vector_invert[1,2] = -n_vector[0]
        n_vector_invert[2,0] = -n_vector[1]
        n_vector_invert[2,1] = n_vector[0]
        
        #print(sita)
        #print(n_vector_invert)
        
        I_M = np.matlib.identity(3)
        #print(I_M)
        # 核心公式：见上图
        R_w2c = I_M + math.sin(sita)*n_vector_invert + n_vector_invert@(n_vector_invert)*(1-math.cos(sita))
        
        R_1 = b.centre - a.centre
        R_w2c_sq = np.squeeze(np.asarray(R_w2c))
        
        R = np.matmul(R_w2c_sq , R_1)
        
        
    radius2 = R.dot(R)
    radius = np.sqrt(radius2)
    xi = a.alpha * b.alpha /(a.alpha + b.alpha)
    lague_x = xi*radius2

    
    l1 = a.l
    l2 = b.l
    
    m1set = np.arange(-l1,l1+1,1)
    m2set = np.arange(-l2,l2+1,1)
    #m1 = a.m
    #m2 = b.m
    
    
    
    I = 0
    
    #F = 0
    #K = 0
    
    for m1 in m1set:
        m1 = m1.item()
        for m2 in m2set:
            
            m2 = m2.item()
            if m1!=0 or m2 !=0: continue
            #if m1==0 or m2 ==0: continue
            #if m1!= -1 or m2 != -1: continue
            #if m1 != m2 : continue # When sum the P orbital, overlap with 
                                  # different orbital cancled, need this condition
          
                
            m = m2 - m1
            

            
            # for one centre overlap integrals
            if radius == 0:
                if l1 == l2 and  m1 == m2: 
                    
                    I = (-1)**l2 * special.gamma(l2+3/2)* (4*xi)**(l2+3/2) /(2*(2*np.pi)**(3/2))
            else:
            # for two centre overlap integrals
                
                theta   =  np.arccos(R[2]/radius)
                phi     =  np.arctan2(R[1],R[0])
                
                # set the range of theta and phi for 
                if theta < 0:
                    theta = theta + 2*np.pi
                if phi < 0:
                    phi = phi + 2*np.pi
                    
                    # use the selection rule to 
                lset = []
                for value in range(abs(l1-l2),l1+l2+1):
                    if (l1+l2+ value) %2 == 0:
                        lset.append(value)
 
                # Sum I for each L
                for l in lset:    
                    if abs(m) > l: continue
                
                    # Calculate the overlap
                    n             = (l1+l2-l)/2
                    C_A_nl        = 2**n * np.math.factorial(n) * (2*xi)**(n+l+3/2)
                    Laguerre      = special.assoc_laguerre(lague_x, n, l+1/2)
                    SolidHarmonic = radius**l * special.sph_harm(m, l, phi, theta)
                    Psi_xi_R      = np.exp(-lague_x)*Laguerre* SolidHarmonic   
                    gaunt_value   = float((-1.0)**m2 *  gaunt(l2,l1,l,-m2,m1,m))
                    
                    #print(CoeFF)
                    
                    I             += ( (-1)**n * gaunt_value * C_A_nl * Psi_xi_R)
                    
                    
                

            
    # Normalized version               
    S = (-1.0)**l2 * (2*np.pi)**(3/2)* Normalize(1/(4*a.alpha),l1)* Normalize(1/(4*b.alpha),l2)*I
    
    #S = (-1.0)**l2 * (2*np.pi)**(3/2)* I
    
    c.volume = S.real * (4/3) * np.pi* (radius**3) *0.05
    #c.volume = S.real
    c.n = a.n + b.n
    c.l = 0
    c.direc = np.array([1,1,1])
    

    
    
    return c


def SHoverlap_volume(a = AtomGaussian(), b = AtomGaussian()):
    

    
    direc_1 = a.direc
    direc_2 = b.direc
    
    com_direc = np.array([direc_1[0]*direc_2[0] , direc_1[1]*direc_2[1], direc_1[2]*direc_2[2]])
    co_mid = np.linalg.norm(com_direc)
    com_direc_norm = np.array([com_direc[0]/co_mid, com_direc[1]/co_mid, com_direc[2]/co_mid])
    
    #print(com_direc_norm)
    z_axis = np.array([0,0,1])

    # T_location 是目标向量
    T_location_norm = com_direc_norm
    
	# originVector是原始向量
    originVector = z_axis
    
    
    
    if (T_location_norm == originVector).all():
        R = b.centre - a.centre
        
        
    else:
        #print(T_location_norm)
        # @是向量点乘
        sita = math.acos(T_location_norm@originVector)
        n_vector_1 = np.cross(T_location_norm ,originVector )
        n_vec_norm = np.linalg.norm(n_vector_1)
        n_vector = np.array([n_vector_1[0]/n_vec_norm , n_vector_1[1]/n_vec_norm, n_vector_1[2]/n_vec_norm])
        
        #n_vector_invert = Matrix((
        #[0,-n_vector[2],n_vector[1]],
        #[n_vector[2],0,-n_vector[0]],
        #[-n_vector[1],n_vector[0],0]
        #))
        n_vector_invert = np.ndarray([3,3])
        n_vector_invert[0,0] = n_vector_invert[1,1] = n_vector_invert[2,2] = 0
        n_vector_invert[0,1] = -n_vector[2]
        n_vector_invert[0,2] = n_vector[1]
        n_vector_invert[1,0] = n_vector[2]
        n_vector_invert[1,2] = -n_vector[0]
        n_vector_invert[2,0] = -n_vector[1]
        n_vector_invert[2,1] = n_vector[0]
        
        #print(sita)
        #print(n_vector_invert)
        
        I_M = np.matlib.identity(3)
        #print(I_M)
        # 核心公式：见上图
        R_w2c = I_M + math.sin(sita)*n_vector_invert + n_vector_invert@(n_vector_invert)*(1-math.cos(sita))
        
        R_1 = b.centre - a.centre
        R_w2c_sq = np.squeeze(np.asarray(R_w2c))
        
        R = np.matmul(R_w2c_sq , R_1)
        
        
    radius2 = R.dot(R)
    radius = np.sqrt(radius2)
    xi = a.alpha * b.alpha /(a.alpha + b.alpha)
    lague_x = xi*radius2

    
    l1 = a.l
    l2 = b.l
    
    m1set = np.arange(-l1,l1+1,1)
    m2set = np.arange(-l2,l2+1,1)
    #m1 = a.m
    #m2 = b.m
    
    
    
    I = 0
    
    #F = 0
    #K = 0
    
    for m1 in m1set:
        m1 = m1.item()
        for m2 in m2set:
            
            m2 = m2.item()
            if m1!=0 or m2 !=0: continue
            #if m1==0 or m2 ==0: continue
            #if m1!= -1 or m2 != -1: continue
            #if m1 != m2 : continue # When sum the P orbital, overlap with 
                                  # different orbital cancled, need this condition
          
                
            m = m2 - m1
            
            
            # for one centre overlap integrals
            if radius == 0:
                if l1 == l2 and  m1 == m2: 
                    
                    I = (-1)**l2 * special.gamma(l2+3/2)* (4*xi)**(l2+3/2) /(2*(2*np.pi)**(3/2))
            else:
            # for two centre overlap integrals
                
                theta   =  np.arccos(R[2]/radius)
                phi     =  np.arctan2(R[1],R[0])
                
                # set the range of theta and phi for 
                if theta < 0:
                    theta = theta + 2*np.pi
                if phi < 0:
                    phi = phi + 2*np.pi
                    
                    # use the selection rule to 
                lset = []
                for value in range(abs(l1-l2),l1+l2+1):
                    if (l1+l2+ value) %2 == 0:
                        lset.append(value)
 
                # Sum I for each L
                for l in lset:    
                    if abs(m) > l: continue
                
                    # Calculate the overlap
                    n             = (l1+l2-l)/2
                    C_A_nl        = 2**n * np.math.factorial(n) * (2*xi)**(n+l+3/2)
                    Laguerre      = special.assoc_laguerre(lague_x, n, l+1/2)
                    SolidHarmonic = radius**l * special.sph_harm(m, l, phi, theta)
                    Psi_xi_R      = np.exp(-lague_x)*Laguerre* SolidHarmonic   
                    gaunt_value   = float((-1.0)**m2 *  gaunt(l2,l1,l,-m2,m1,m))
                    
                    #print(CoeFF)
                    
                    I             += ( (-1)**n * gaunt_value * C_A_nl * Psi_xi_R)

            
    # Normalized version               
    S = (-1.0)**l2 * (2*np.pi)**(3/2)* Normalize(1/(4*a.alpha),l1)* Normalize(1/(4*b.alpha),l2)*I
    #Unnormalized version
    #S = (-1.0)**l2 * (2*np.pi)**(3/2)* I
    #S_volume = S.real * (4/3) * np.pi* (radius**3) *0.05
    
    
    if l1==l2==1 and radius==0:
        S_volume = 7*S.real
    else:
        S_volume = S.real * (4/3) * np.pi* (radius**3) *0.05
        

    #if l1==1 or l2==1:
     #   
      #  if l1==l2:
       #     if radius ==0:
        #        S_volume = a.volume*S.real
         #   else:
          #      S_volume = S.real * (4/3) * np.pi* (radius**3) *0.05
        #
        #else:
         #   S_volume = S.real * (4/3) * np.pi* (radius**3) *0.05
    #else:
     #   S_volume = S.real * (4/3) * np.pi* (radius**3) *0.05

    return S.real
     
    

def Gradient(n, l, m, alpha, r_vec):
    '''The function takes input n, l, m from the resulted overlap function
        AKA Phi^b_nlm'''
    #transform to spherical polar coordinate
    r2 = r_vec.dot(r_vec)
    r = np.sqrt(r2)
    theta   =  np.arccos(r_vec[2]/r)
    phi     =  np.arctan2(r_vec[1],r_vec[0])
    
    if theta < 0:
        theta = theta + 2*np.pi
    if phi < 0:
        phi = phi + 2*np.pi
    
    #calculate functions that only related to the radius
    exp = np.exp(-alpha*r2)
    f = r**l *exp* special.assoc_laguerre(alpha*r2, n, l+1/2)
    df = (l/r - 2*alpha*r) * f \
        - 2*alpha* exp *r**(l+1) * special.assoc_laguerre(alpha*r2, n-1, l+3/2)
    #df = l* r**(l-1) * special.assoc_laguerre(alpha*r**2, n, l+1/2)\
        #- 2*alpha*r**(l+1)*special.assoc_laguerre(alpha*r**2, n-1, l+3/2)
    
    #This part is same for all direction, so evalute outside the loop
    F_plus = np.sqrt((l+1)/(2*l+3)) * (df - l*f/r)
    F_minus = np.sqrt(l/(2*l-1)) * (df + (l+1)*f/r)
            
    #Define the transformation matrix from spherical basis to cartizian basis
    U = np.zeros([3,3]).astype(complex)   
    U[0,0] = -1/np.sqrt(2)
    U[0,1] = 1/np.sqrt(2)
    U[1,0] = U[1,1] = complex(0,1/np.sqrt(2)) #!!!The sign should be minus
                                            #But plus give the right answer?
    U[2,2] = 1
    
    G_spherical = np.zeros(3).astype(complex)
    it = 0
    for mu in [+1,-1,0]:  
        
        G_plus = clebsch_gordan(l,1,l+1,m,mu,m+mu)\
            * np.nan_to_num(special.sph_harm(m+mu,l+1, phi, theta)) * F_plus
            
        G_minus = clebsch_gordan(l,1,l-1,m,mu,m+mu)\
            * np.nan_to_num(special.sph_harm(m+mu,l-1, phi, theta)) * F_minus
            
        G_spherical[it] = G_plus - G_minus
                        
        it+=1
    G= np.matmul(U,G_spherical)
    return G

def test_F(n, l, m, alpha, r_vec):
    
    r2 = r_vec.dot(r_vec)
    r = np.sqrt(r2)
    theta   =  np.arccos(r_vec[2]/r)
    phi     =  np.arctan2(r_vec[1],r_vec[0])
    if theta < 0:
        theta = theta + 2*np.pi
    if phi < 0:
        phi = phi + 2*np.pi
    f = r**l *np.exp(-alpha*r2)* special.assoc_laguerre(alpha*r2, n, l+1/2)
    Y = np.nan_to_num(special.sph_harm(m,l, phi, theta))
    
    return f*Y






#%%
