# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:12:45 2020

@author: THINKPAD
"""

import numpy as np
from scipy.optimize import differential_evolution as DE
#from openbabel import openbabel
from rdkit import Chem
from AlignmentInfo import AlignmentInfo
#from GaussianVolume import GaussianVolume, Molecule_volume , getScore
from GaussianVolume import GaussianVolume
#from AtomGaussian import atomIntersection

from ShapeAlignment import ShapeAlignment
from SolutionInfo import SolutionInfo, updateSolutionInfo

import moleculeRotation 

from collections import deque
maxIter = 0
MATRIXMAP = 0


#%%
refMol = Chem.MolFromMolFile('ref.mol')



#refMol = Chem.MolFromSmiles('NS(=O)(=O)c1ccc(C(=O)N2Cc3ccccc3C(c3ccccc3)C2)cc1')
#refMol = Chem.MolFromMolFile('sangetan.mol')
#pre_refMol = Chem.MolFromSmiles('COc1ccc(-c2nc3c4ccccc4ccc3n2C(C)C)cc1')   
#pre_refMol = Chem.MolFromMolFile('AAR.mol')
#pre_refMol_H=Chem.AddHs(pre_refMol)
#AllChem.EmbedMolecule(pre_refMol_H) 
#AllChem.MMFFOptimizeMolecule(pre_refMol_H)
#refMol = Chem.RemoveHs(pre_refMol_H)


refVolume = GaussianVolume()

# List all Gaussians and their respective intersections
Molecule_volume(refMol,refVolume)

# 	Move the Gaussian towards its center of geometry and align with principal axes
#initOrientation(refVolume)

#%%

# dodge all scoreonly process

#Create a class to hold the best solution of an iteration
bestSolution = SolutionInfo()
bestSolution.refAtomVolume = refVolume.overlap
bestSolution.refCenter = refVolume.centroid
bestSolution.refRotation = refVolume.rotation

# Create the set of Gaussians of database molecule
inf = open('DK+clean.sdf','rb')#
fsuppl = Chem.ForwardSDMolSupplier(inf)
Molcount = 0
SolutionTable = np.zeros([101,3])
for dbMol in fsuppl: 
    Molcount+=1
    #if Molcount != 1: continue
    #if dbMol is None: continue

    dbName = dbMol.GetProp('zinc_id')
    
    dbVolume = GaussianVolume()
    Molecule_volume(dbMol,dbVolume)
    
    res = AlignmentInfo()
    bestScore = 0.0
    
    initOrientation(dbVolume)
    aligner = ShapeAlignment(refVolume,dbVolume)
    aligner.setMaxIterations(maxIter) 
    
    for l in range(4):
        
        quat = np.zeros(4)
        quat[l] = 1.0
        nextRes = aligner.gradientAscent(quat)
        checkVolumes(refVolume, dbVolume, nextRes)
        ss = getScore('tanimoto', nextRes.overlap, refVolume.overlap, dbVolume.overlap)

        if ss > bestScore:
            
            res = nextRes
            bestScore = ss
                   
        if bestScore > 0.98:  continue


    if maxIter > 0:
        nextRes = aligner.simulatedAnnealing(res.rotor)
        checkVolumes(refVolume, dbVolume, nextRes)
        ss = getScore('tanimoto', nextRes.overlap, refVolume.overlap, dbVolume.overlap)
        if (ss > bestScore):
            bestScore = ss
            res = nextRes
    dbVolume.gaussians.clear()
    dbVolume.levels.clear()
    dbVolume.childOverlaps.clear()
    print(res.rotor)             
    updateSolutionInfo(bestSolution, res, bestScore, dbVolume)
    bestSolution.dbMol = dbMol
    bestSolution.dbName = dbName  
    
    if bestSolution.score > 0.7: 
        
        ''''''post-process molecules''''''
        
        # Translate and rotate the molecule towards its centroid and inertia axes
        moleculeRotation.positionMolecule(bestSolution.dbMol, bestSolution.dbCenter, bestSolution.dbRotation)
                
       	# Rotate molecule with the optimal
        moleculeRotation.rotateMolecule(bestSolution.dbMol, bestSolution.rotor)
       
       	# Rotate and translate the molecule with the inverse rotation and translation of the reference molecule
        moleculeRotation.repositionMolecule(bestSolution.dbMol,refVolume.centroid, refVolume.rotation )
           

    SolutionTable[Molcount] = np.array([bestSolution.score,bestSolution.atomOverlap,bestSolution.dbAtomVolume])
#%%
#test = []
#for sublist in test1:
    #test.append([item for item in sublist])

np.savetxt("foo.csv", SolutionTable, delimiter=",")
#np.savetxt("test1.csv", test1, delimiter=",")
#np.savetxt("test2.txt", test2, delimiter=",",fmt='%1.6f')

#%%
imported_db =  Chem.MolToMolBlock(bestSolution.dbMol,confId=-1)    
imported_ref = Chem.MolToMolBlock(refMol,confId=-1) 
with open(r"C:\Users\THINKPAD\Desktop\Msci project\Msci_project_code\Msci_project\Data\db.mol", "w") as newfile:
       newfile.write(imported_db)
       
with open(r"C:\Users\THINKPAD\Desktop\Msci project\Msci_project_code\Msci_project\Data\ref.mol", "w") as newfile:
       newfile.write(imported_ref)
       

#%%
       
refMol = Chem.MolFromMolFile('ref.mol')
refVolume = GaussianVolume()
Molecule_volume(refMol,refVolume)
dbMol = Chem.MolFromMolFile('GAR.mol')
dbVolume = GaussianVolume()
Molecule_volume(dbMol,dbVolume)
initOrientation(dbVolume)


aligner1 = ShapeAlignment(refVolume,dbVolume)
nextRes = aligner1.BlackBox_function()

ss = getScore('tanimoto', nextRes.overlap, refVolume.overlap, dbVolume.overlap)
print(ss)    


#%% Current Black-Box test code!!!!!!!!!!!!!!!!!!!!!!!!!!



refMol = Chem.MolFromMolFile('test_1.mol')
refVolume = GaussianVolume()
Molecule_volume(refMol,refVolume)
initOrientation(refVolume)



dbMol = Chem.MolFromMolFile('test_2.mol')
dbVolume = GaussianVolume()
Molecule_volume(dbMol,dbVolume)
initOrientation(dbVolume)



bestSolution = SolutionInfo()
bestSolution.refAtomVolume = refVolume.overlap
bestSolution.refCenter = refVolume.centroid
bestSolution.refRotation = refVolume.rotation


ro, overlapvolume = BB_function(refVolume, dbVolume)
print(ro)
print(overlapvolume)


res = AlignmentInfo()


res.rotor = ro
res.overlap = overlapvolume

checkVolumes(refVolume, dbVolume, res)
print(res.overlap)
ss = getScore('tanimoto', res.overlap, refVolume.overlap, dbVolume.overlap)
print(ss)

updateSolutionInfo(bestSolution, res, ss, dbVolume)
bestSolution.dbMol = dbMol
#bestSolution.dbName = dbName  


moleculeRotation.positionMolecule(bestSolution.dbMol, bestSolution.dbCenter, bestSolution.dbRotation)
                
       	# Rotate molecule with the optimal
moleculeRotation.rotateMolecule(bestSolution.dbMol, bestSolution.rotor)
       
       	# Rotate and translate the molecule with the inverse rotation and translation of the reference molecule
moleculeRotation.repositionMolecule(bestSolution.dbMol,refVolume.centroid, refVolume.rotation )


imported_db =  Chem.MolToMolBlock(bestSolution.dbMol,confId=-1)    
imported_ref = Chem.MolToMolBlock(refMol,confId=-1) 
with open(r"C:\Users\86153\dbmol.mol", "w") as newfile:
       newfile.write(imported_db)
       
with open(r"C:\Users\86153\refmol.mol", "w") as newfile:
       newfile.write(imported_ref)


#%%
       
refMol = Chem.MolFromMolFile('ref.mol')
refVolume = GaussianVolume()
Molecule_volume(refMol,refVolume)
dbMol = Chem.MolFromMolFile('GAR.mol')
dbVolume = GaussianVolume()
Molecule_volume(dbMol,dbVolume)
initOrientation(dbVolume)

aligner = ShapeAlignment(refVolume, dbVolume)
bestScore = 0.0
for l in range(4):
        
        quat = np.zeros(4)
        quat[l] = 1.0
        nextRes = aligner.gradientAscent(quat)
        checkVolumes(refVolume, dbVolume, nextRes)
        ss = getScore('tanimoto', nextRes.overlap, refVolume.overlap, dbVolume.overlap)

        if ss > bestScore:
            
            res = nextRes
            bestScore = ss
                   
            if bestScore > 0.98:  continue

print(refVolume.overlap)
print(dbVolume.overlap)
print(nextRes.overlap)
#ss = getScore('tanimoto', nextRes.overlap, refVolume.overlap, dbVolume.overlap)
print(ss)    
