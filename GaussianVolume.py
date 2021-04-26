# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:43:22 2020

@author: THINKPAD
"""

import numpy as np
from collections import deque
from rdkit import Chem
from AlignmentInfo import AlignmentInfo
from AtomGaussian import AtomGaussian
#from openbabel import openbabel as ob
from scipy.optimize import differential_evolution as DE

class GaussianVolume(AtomGaussian):
    
    def __init__(self):
        
        self.volume = 0.0
        self.overlap = 0.0
        self.centroid = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])
        
        # Store the original atom gaussians and the overlap gaussians calculated later
        self.gaussians = []
        
        # store the overlap tree used for molecule molecule intersection
        self.childOverlaps = [] 
        
        # Store the number of gaussians for each level
        self.levels = []
                
def GAlpha(atomicnumber): #returns the Alpha value of the atom
    
        switcher={
   1: 1.679158285,
     
   3: 0.729980658,
     
   5: 0.604496983,
     
   6: 0.836674025,
     
   7: 1.006446589,
     
   8: 1.046566798,
     
   9: 1.118972618,
     
   11: 0.469247983,
     
   12: 0.807908026,
     
   14: 0.548296583,
     
   15: 0.746292571,
    
   16: 0.746292571,
     
   17: 0.789547080,
     
   19: 0.319733941,
     
   20: 0.604496983,
     
   26: 1.998337133,
     
   29: 1.233667312,
     
   30: 1.251481772,
     
   35: 0.706497569,
     
   53: 0.616770720,
         }
        return switcher.get(atomicnumber,1.074661303) 

#%%

def Molecule_volume(mol = Chem.rdchem.Mol(),  gv = GaussianVolume()):
    
    EPS = 0.03
    #N = mol.GetNumAtoms()
    B = mol.GetNumBonds()
    
    N_p = 2*B
    
    
    for i in range(N_p):
        i +=1
        gv.childOverlaps.append([])
        gv.gaussians.append(AtomGaussian())
            
    gv.levels.append(N_p)
    gv.volume = 0.0
    gv.centroid = np.array([0.0, 0.0, 0.0])
    
    # Stores the parents of gv.gaussians[i] inside parents[i]
    parents = [[] for i in range(N_p)]

    # Stores the atom index that have intersection with i_th gaussians inside overlaps[i]
    overlaps = [set() for i in range(N_p)] 
  
    #bondIndex = 0 #Bond index actually
    vecIndex = N_p #Used to indicated the initial position of the child gaussian
    
    guassian_weight = 2.828427125 
    
    conf = mol.GetConformer()
    
    pair_index = 0
    
    
    for bond in mol.GetBonds():
        
        douible_check_idx = 2
        
        while douible_check_idx != 0:
            
            
            if (pair_index%2) == 0:
                atom = bond.GetBeginAtom()
                atom_idx = atom.GetIdx()
                douible_check_idx -=1
                atom_num = atom.GetAtomicNum()
                
                atom_2 = bond.GetEndAtom()
                atom_idx_2 = atom_2.GetIdx()
             
                
            else:
                atom = bond.GetEndAtom()
                atom_idx = atom.GetIdx()
                douible_check_idx -=1
                atom_num = atom.GetAtomicNum()
                          
                
                atom_2 = bond.GetBeginAtom()
                atom_idx_2 = atom_2.GetIdx()
                
            
            if atom_num == 1: continue 
            #if right_atom.GetAtomicNum() == 1: continue 
            atom_num = atom.GetAtomicNum()
            gv.gaussians[pair_index].centre = np.array(conf.GetAtomPosition(atom_idx))
            
            # value checked, same with mol file
            gv.gaussians[pair_index].alpha = GAlpha(atom_num)
            gv.gaussians[pair_index].weight = guassian_weight 
            gv.gaussians[pair_index].l = 1
            gv.gaussians[pair_index].direc = np.array(conf.GetAtomPosition(atom_idx_2))- np.array(conf.GetAtomPosition(atom_idx))
            radius_VDW = Chem.GetPeriodicTable().GetRvdw(atom_num)
            
            #radius_VDW =  ob.GetVdwRad(atom.GetAtomicNum())
            '''it looks like the GetRvdw function in rdkit give 1.95 for Carbon, 
            which is the vdw radius for Br in our paper, here I redefined the value'''
            gv.gaussians[pair_index].volume = (1/3) * (4.0 * np.pi/3.0) * radius_VDW **3
            
            #!!!Apply one third
            #checked, give the same value as (np.pi/gv.gaussians[atomIndex].alpha)**1.5 * gv.gaussians[atomIndex].weight
            gv.gaussians[pair_index].n = 1 
            
            
            '''Update volume and centroid of the Molecule'''
            gv.volume += gv.gaussians[pair_index].volume
            gv.centroid += gv.gaussians[pair_index].volume*gv.gaussians[pair_index].centre
        
            '''loop over every atom to find the second level overlap'''
            
            for i in range(pair_index):
                
                ga = SHoverlap(gv.gaussians[i], gv.gaussians[pair_index])
                     
                # Check if overlap is sufficient enough
 
                if ga.volume / (gv.gaussians[i].volume + gv.gaussians[pair_index].volume - ga.volume) < EPS: continue
            
                gv.gaussians.append(ga) 
                gv.childOverlaps.append([]) 
                
                #append a empty list in the end to store child of this overlap gaussian
                parents.append([i,pair_index])
                overlaps.append(set())
                
                gv.volume -=  ga.volume
                gv.centroid -=   ga.volume * ga.centre
                
                overlaps[i].add(pair_index)                  
                # store the position of the child (vecIndex) in the root (i)   
                gv.childOverlaps[i].append(vecIndex) 
                
                vecIndex+=1
                
            pair_index += 1
       

    
    startLevel = pair_index
    nextLevel = len(gv.gaussians)
    gv.levels.append(nextLevel)
        
    
    LEVEL = 6 
    
    for l in range(2,LEVEL):
        for i in range(startLevel,nextLevel):

            # parents[i] is a pair list e.g.[a1,a2]
            a1 = parents[i][0]
            a2 = parents[i][1]
            
            # find elements that overlaps with both gaussians(a1 and a2)
            overlaps[i] = overlaps[a1] & overlaps[a2]
 

            if len(overlaps[i]) == 0: continue
            for elements in overlaps[i]:
                
                
                # check if there is a wrong index
                if elements <= a2: continue
               
                ga = SHoverlap(gv.gaussians[i],gv.gaussians[elements])
                

            
                if ga.volume/(gv.gaussians[i].volume + gv.gaussians[elements].volume - ga.volume) < EPS: continue
                    
                gv.gaussians.append(ga)
                #append a empty list in the end to store child of this overlap gaussian
                gv.childOverlaps.append([]) 
                
                parents.append([i,elements])
                overlaps.append(set())
                
                if (ga.n % 2) == 0:# even number overlaps give positive contribution
                    gv.volume -=  ga.volume
                    gv.centroid -=   ga.volume*ga.centre
                else:         # odd number overlaps give negative contribution
                    gv.volume +=  ga.volume
                    gv.centroid +=  ga.volume*ga.centre
                
                # store the position of the child (vecIndex) in the root (i)
                gv.childOverlaps[i].append(vecIndex) 
                
                vecIndex+=1
        

        
        startLevel = nextLevel
        nextLevel = len(gv.gaussians)
        gv.levels.append(nextLevel)

    
    overlaps.clear()#!!! why so complacated in C++ code?
    
    parents.clear()
    gv.overlap = Molecule_overlap(gv,gv)
    
    return gv


#%%
'''Build up the mass matrix'''
def initOrientation(gv = GaussianVolume()):
    mass_matrix = np.zeros(shape=(3,3))
    iu = np.triu_indices(3)
    iu2 = np.triu_indices(3,k=1)
    il2 = np.tril_indices(3, k=-1)

    gv.centroid /= gv.volume #normalise the centroid
    

    for i in gv.gaussians:
        i.centre -= gv.centroid
        centre = i.centre[:,None]
        outer = np.matmul(centre,centre.T)
        if i.n % 2 == 0: # for even number of atom, negative contribution
        
            mass_matrix[iu] -= i.volume * outer[iu]

        else:            # for odd number of atom, positive contribution
            mass_matrix[iu] += i.volume * outer[iu]

        
    # set lower triangle due to its sysmetry  
    mass_matrix[il2] = mass_matrix[iu2]

    #normalise
    mass_matrix /= gv.volume
    
    #print('mass_matrix')
    #print(mass_matrix)
    
    #singular value decomposition
    gv.rotation, s, vh = np.linalg.svd(mass_matrix, compute_uv=True)
    #gv.rotation[:,1] = -gv.rotation[:,1]
    #gv.rotation[:,0] = -gv.rotation[:,0]
  
    #project the atoms' coordinates onto the principle axes
    if np.linalg.det(gv.rotation) < 0:
        gv.rotation[:,2] = -gv.rotation[:,2]
        
    for i in gv.gaussians:
        i.centre = np.einsum('ij,i->j',gv.rotation,i.centre) #!!! not matrix multiplication
        #i.centre = gv.rotation.dot(i.centre)
    
    isotropic = np.array([1,1,1])
    
    for i in range(len(gv.gaussians)):
        
        if (gv.gaussians[i].direc == isotropic).all():
            gv.gaussians[i].direc = isotropic
            
        else:
            
            if (i%2) == 0:
                gv.gaussians[i].direc = gv.gaussians[i+1].centre - gv.gaussians[i].centre
            else:
                gv.gaussians[i].direc = gv.gaussians[i-1].centre - gv.gaussians[i].centre
        
              
    return gv


#%%

def Molecule_overlap(gRef = GaussianVolume(), gDb = GaussianVolume()):
    processQueue= deque() #Stores the pair of gaussians that needed to calculate overlap
    overlap_volume = 0
    
    N1 = gRef.levels[0] #Reference molecule
    N2 = gDb.levels[0] #Database molecule
    
    '''loop over the atoms in both molecules'''
    EPS = 0.03
    for i in range(N1):
        for j in range(N2):
            
            #Cij = gRef.gaussians[i].alpha * gDb.gaussians[j].alpha / (gRef.gaussians[i].alpha + gDb.gaussians[j].alpha)
            			
            # Variables to store sum and difference of components
            #d = (gRef.gaussians[i].centre - gDb.gaussians[j].centre)
            #d_sqr = d.dot(d)
           
            # Compute overlap volume
            
            #V_ij = gRef.gaussians[i].weight * gDb.gaussians[j].weight * (np.pi/(gRef.gaussians[i].alpha + gDb.gaussians[j].alpha))**1.5 * np.exp(- Cij * (d_sqr )) 
            
            V_ij = SHoverlap_volume(gRef.gaussians[i],gDb.gaussians[j])
            

            if V_ij / (gRef.gaussians[i].volume + gDb.gaussians[j].volume - V_ij) < EPS: continue
                
            overlap_volume += V_ij
           
            # Loop over child nodes and add to queue1
            d1 = gRef.childOverlaps[i]
            d2 = gDb.childOverlaps[j]
           
            # First add (i,child(j))
            if d2 != None:
                for it2 in d2:
                    processQueue.append([i,it2])
        
            #Second add (child(i),j)
            if d1!= None:
                for it1 in d1:
                    processQueue.append([it1,j])
    
    while len(processQueue) != 0: # loop when processQueue is not empty
    
        pair = processQueue.popleft()
               
        i = pair[0]
        j = pair[1]
        
        #Cij = gRef.gaussians[i].alpha * gDb.gaussians[j].alpha / (gRef.gaussians[i].alpha + gDb.gaussians[j].alpha)
            			
        # Variables to store sum and difference of components
        #d = (gRef.gaussians[i].centre - gDb.gaussians[j].centre)
        #d_sqr = d.dot(d)
        			
        # Compute overlap volume
        #V_ij = gRef.gaussians[i].weight * gDb.gaussians[j].weight * (np.pi/(gRef.gaussians[i].alpha + gDb.gaussians[j].alpha))**1.5 * np.exp(- Cij * (d_sqr )) 
        
        V_ij = SHoverlap_volume(gRef.gaussians[i],gDb.gaussians[j])
        

        if V_ij / (gRef.gaussians[i].volume + gDb.gaussians[j].volume - V_ij) < EPS: continue
                           
        if ((gRef.gaussians[i].n + gDb.gaussians[j].n)%2) == 0: 
            
            overlap_volume += V_ij
        else:
            overlap_volume -= V_ij
            
        d1 = gRef.childOverlaps[i]
        d2 = gDb.childOverlaps[j]
        
        
        if d1 != None and gRef.gaussians[i].n >  gDb.gaussians[j].n:
             #Add (child(i),j)
            for it1 in d1:
                processQueue.append([it1,j])
                
        else: 
            if d2 != None:
                # add (i,child(j))
                for it2 in d2:
                    processQueue.append([i,it2])
            if d1 != None and gDb.gaussians[j].n - gRef.gaussians[i].n < 2:
                # add (child(i),j)
                for it1 in d1:           
                    processQueue.append([it1,j])
                    
    return overlap_volume
                  
def getScore(name, Voa, Vra, Vda):
    
    if name == 'tanimoto':
        #print('Voa_' + str(Voa) +'Vra_' + str(Vra) + 'Vda_' + str(Vda))
        return Voa/(Vra+Vda-Voa)
    elif name == 'tversky_ref':
        return Voa / (0.95*Vra + 0.05*Vda)
    elif name == 'tversky_db':
        return Voa/(0.05*Vra+0.95*Vda)
    
    return 0.0


def checkVolumes(gRef = GaussianVolume, gDb = GaussianVolume,
                 res = AlignmentInfo()):
    
    if res.overlap > gRef.overlap:
        res.overlap = gRef.overlap
        
    if res.overlap > gDb.overlap:
        res.overlap = gDb.overlap
        
    return 




def BB_function(gRef = GaussianVolume(), gDb = GaussianVolume()):
    
    
        
    
    def inside_optimize(x) :
        
        
        
        EPS = 0.03

        R_x = np.ndarray([3,3])  
        R_x[0,0] = 1
        R_x[1,0] = R_x[0,1] = R_x[2,0] = R_x[0,2] = 0
        R_x[1,1] = R_x[2,2] = np.cos(x[0])
        R_x[1,2] = np.sin(x[0])
        R_x[2,1] = - R_x[1,2]
        
        R_y = np.ndarray([3,3])  
        R_y[1,1] = 1
        R_y[1,0] = R_y[1,2] = R_y[0,1] = R_y[2,1] = 0
        R_y[0,0] = R_y[2,2] = np.cos(x[1])
        R_y[2,0] = np.sin(x[1])
        R_y[0,2] = - R_y[2,0]
        
        R_z = np.ndarray([3,3])  
        R_z[2,2] = 1
        R_z[2,0] = R_z[2,1] = R_z[0,2] = R_z[1,2] = 0
        R_z[0,0] = R_z[1,1] = np.cos(x[2])
        R_z[0,1] = np.sin(x[2])
        R_z[1,0] = - R_z[0,1]
        
        R_matrix_1 = np.matmul(R_x , R_y)
        R_matrix = np.matmul(R_matrix_1 , R_z)
        
        
        N1 = gRef.levels[0] #Reference molecule
        N2 = gDb.levels[0] #Database molecule
        
        
        db_centre_list = []
        db_direc_list = []
        db_alpha_list = []
        db_l_list = []
        isotropic = np.array([1,1,1])
        
        temporary_atom = AtomGaussian()
        #Rotate all the atoms in database molecule first:
        for i in gDb.gaussians:
            
            db_centre_list.append( np.matmul( R_matrix, i.centre))
        
        for i in gDb.gaussians:
            
            db_alpha_list.append( i.alpha)
            
        for i in gDb.gaussians:
            
            db_l_list.append( i.l)
        
        for i in range(len(gDb.gaussians)):
        
            if (gDb.gaussians[i].direc == isotropic).all(): 
                
                db_direc_list.append(isotropic)
            else:
                
                if (i%2) == 0:
                    db_direc_list.append(gDb.gaussians[i+1].centre - gDb.gaussians[i].centre)
                else:
                    db_direc_list.append(gDb.gaussians[i-1].centre - gDb.gaussians[i].centre)
            
        
        processQueue= deque() #Stores the pair of gaussians that needed to calculate overlap
        overlap_volume = 0
        
        d1 = []
        d2 = []
        #N1 = VAR._gRef.levels[0] #Reference molecule
        #N2 = VAR._gDb.levels[0] #Database molecule
        '''loop over the atoms in both molecules'''
    
        for i in range(N1):
            for j in range(N2):
        
                
                
                #Cij = gRef.gaussians[i].alpha * gDb.gaussians[j].alpha / (gRef.gaussians[i].alpha + gDb.gaussians[j].alpha)
            			
                # Variables to store sum and difference of components
                #d = (gRef.gaussians[i].centre - db_centre_list[j])
                #d_sqr = d.dot(d)
        			
                # Compute overlap volume
                #Vij = gRef.gaussians[i].weight * gDb.gaussians[j].weight * (np.pi/(gRef.gaussians[i].alpha + gDb.gaussians[j].alpha))**1.5 * np.exp(- Cij * (d_sqr )) 
                #Vij = atomIntersection(gRef.gaussians[i], gDb.gaussians[j])
                
                temporary_atom.centre = db_centre_list[j]
                temporary_atom.direc = db_direc_list[j]
                temporary_atom.alpha = db_alpha_list[j]
                temporary_atom.l = db_l_list[j]
                
                Vij = SHoverlap_volume(gRef.gaussians[i],temporary_atom)
                
                if Vij / (gRef.gaussians[i].volume + gDb.gaussians[j].volume - Vij) < EPS: continue
                                    
                overlap_volume += Vij
            
                #overlap_volume += V_ij
            
                # Loop over child nodes and add to queue1
                d1 = gRef.childOverlaps[i]
                d2 = gDb.childOverlaps[j]
       
                # First add (i,child(j))
                if d2 != None:
                    for it2 in d2:
                        processQueue.append([i,it2])
    
                #Second add (child(i),j)
                if d1!= None:
                    for it1 in d1:
                        processQueue.append([it1,j])
    
        while len(processQueue) != 0: # loop when processQueue is not empty
    
            pair = processQueue.popleft()
           
            i = pair[0]
            j = pair[1]
    
            
            #Cij = gRef.gaussians[i].alpha * gDb.gaussians[j].alpha / (gRef.gaussians[i].alpha + gDb.gaussians[j].alpha)
            			
            # Variables to store sum and difference of components
            #d = (gRef.gaussians[i].centre - db_centre_list[j])
            #d_sqr = d.dot(d)
        			
            # Compute overlap volume
            #Vij = gRef.gaussians[i].weight * gDb.gaussians[j].weight * (np.pi/(gRef.gaussians[i].alpha + gDb.gaussians[j].alpha))**1.5 * np.exp(- Cij * (d_sqr ))     
            #Vij = atomIntersection(gRef.gaussians[i], gDb.gaussians[j])
            
            temporary_atom.centre = db_centre_list[j]
            temporary_atom.direc = db_direc_list[j]
            temporary_atom.alpha = db_alpha_list[j]
            temporary_atom.l = db_l_list[j]
                
            Vij = SHoverlap_volume(gRef.gaussians[i],temporary_atom)
            
            if Vij / (gRef.gaussians[i].volume + gDb.gaussians[j].volume - Vij) < EPS: continue
                       
            if ((gRef.gaussians[i].n + gDb.gaussians[j].n)%2) == 0: 
        
                #overlap_volume += V_ij
                overlap_volume += Vij
            else:
                #overlap_volume -= V_ij
                overlap_volume -= Vij
            
            d1 = gRef.childOverlaps[i]
            d2 = gDb.childOverlaps[j]
            
            ####SAME HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if d1 != None and gRef.gaussians[i].n >  gDb.gaussians[j].n:
                #Add (child(i),j)
                for it1 in d1:
                    processQueue.append([it1,j])
            
            else: 
                if d2 != None:
                        # add (i,child(j))
                    for it2 in d2:
                        processQueue.append([i,it2])
                if d1 != None and gDb.gaussians[j].n - gRef.gaussians[i].n < 2:
                    # add (child(i),j)
                    for it1 in d1:           
                        processQueue.append([it1,j])
            ####SAME ABOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        return -overlap_volume



    bounds = [(0, 2*np.pi)]*3
    
    result = list(de(inside_optimize, bounds))
    
    
    R_x = np.ndarray([3,3])  
    R_x[0,0] = 1
    R_x[1,0] = R_x[0,1] = R_x[2,0] = R_x[0,2] = 0
    R_x[1,1] = R_x[2,2] = np.cos(result[-1][0][0])
    R_x[1,2] = np.sin(result[-1][0][0])
    R_x[2,1] = - R_x[1,2]
    
    R_y = np.ndarray([3,3])  
    R_y[1,1] = 1
    R_y[1,0] = R_y[1,2] = R_y[0,1] = R_y[2,1] = 0
    R_y[0,0] = R_y[2,2] = np.cos(result[-1][0][1])
    R_y[2,0] = np.sin(result[-1][0][1])
    R_y[0,2] = - R_y[2,0]
    
    R_z = np.ndarray([3,3])  
    R_z[2,2] = 1
    R_z[2,0] = R_z[2,1] = R_z[0,2] = R_z[1,2] = 0
    R_z[0,0] = R_z[1,1] = np.cos(result[-1][0][2])
    R_z[0,1] = np.sin(result[-1][0][2])
    R_z[1,0] = - R_z[0,1]
    
    R_matrix_1 = np.matmul(R_x , R_y)
    R_matrix = np.matmul(R_matrix_1 , R_z)
    
    qw = np.sqrt(1 + R_matrix[0,0] + R_matrix[1,1] + R_matrix[2,2]) / 2
    qx = (R_matrix[2,1] - R_matrix[1,2]) / (4 * qw)
    qy = (R_matrix[0,2] - R_matrix[2,0]) / (4 * qw)
    qz = (R_matrix[1,0] - R_matrix[0,1]) / (4 * qw)
    
    ro = np.array((qw , qx, qy, qz))

    q_value = np.linalg.norm(ro)
    ro_norm = np.array((ro[0]/q_value, ro[1]/q_value, ro[2]/q_value, ro[3]/q_value ))
    
    positive_overlap_vol = -result[-1][1]
    
    return ro_norm ,positive_overlap_vol    
    
        
        
    
    

def de(fobj, bounds, mut=0.8, crossp=0.8, popsize=20, its=30):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]




                
        
            
        
        
        
                       
                   
                    
                   
               