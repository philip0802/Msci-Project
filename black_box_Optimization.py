# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 23:21:31 2021

@author: 86153
"""

from benderclient import Bender
import numpy as np
from ShapeAlignment import ShapeAlignment, BlackBox_function
from rdkit import Chem
from AtomGaussian import AtomGaussian,SHoverlap
from GaussianVolume import GaussianVolume, Molecule_volume, initOrientation
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution

# Initialize Bender



#%%
bender = Bender()



refMol = Chem.MolFromMolFile('benzene.mol')
refVolume = GaussianVolume()
Molecule_volume(refMol,refVolume)
dbMol = Chem.MolFromMolFile('sibianxing.mol')
dbVolume = GaussianVolume()
Molecule_volume(dbMol,dbVolume)
initOrientation(dbVolume)




# Create an experiment
bender.create_experiment(
    name='quaternion experiment',
    description='Find the maximum molecular overlap',
    metrics=[{"metric_name": "quaternion", "type": "reward"}],
)

# Create an algo (here the sinus function with one parameter)
bender.create_algo(
    name='Overlap function',
    hyperparameters=[
        {
            "name": 'w',
            "category": "uniform",
            "search_space": {
                "low": -1,
                "high": 1,
            },
        },
                    
        {
            "name": 'x',
            "category": "uniform",
            "search_space": {
                "low": -1,
                "high": 1,
            },
        },
                    
        {
            "name": 'y',
            "category": "uniform",
            "search_space": {
                "low": -1,
                "high": 1,
            },
        },
                    
         {
            "name": 'z',
            "category": "uniform",
            "search_space": {
                "low": -1,
                "high": 1,
            },
        },            
    ]
)

#test_list=[]
# Ask bender for values to try
for _ in range(20):
    
    overlap_volume = 0
    
    suggestion = bender.suggest(metric="quaternion", optimizer="parzen_estimator")
    ro = np.array((suggestion["w"] , suggestion["x"], suggestion["y"], suggestion["z"]))
    q_value = np.linalg.norm(ro)
    ro_norm = np.array((ro[0]/q_value, ro[1]/q_value, ro[2]/q_value, ro[3]/q_value ))
    
    for i in refVolume.gaussians:
        for j in dbVolume.gaussians:
            
            Aij,A16 = ShapeAlignment._updateMatrixMap(i,j)
    # Get a set of Hyperparameters to test
            

    # Run the sinus function
    #q_value = suggestion["w"]**2 + suggestion["x"]**2 +suggestion["y"]**2 +suggestion["z"]**2
    
           
            
            #while q_value == 1:
        
     
            
        #sinus_value = np.sin(suggestion["x"])
            Aq = np.matmul(Aij, ro_norm) #Calculation of q′Aq, rotor product
            qAq = np.matmul(ro_norm, Aq)                
            Vij = A16 * np.exp( -qAq ) 
            overlap_volume += Vij

        # Feed Bender a trial
    bender.create_trial(
            hyperparameters=suggestion,
            results={"quaternion": overlap_volume}
        )        
            
            
    print("q: ", ro, " value :", overlap_volume)
        
        
       #test_list.append(["x: ", suggestion["x"], " value :", sinus_value])



#%%

#Aij,A16 = ShapeAlignment._updateMatrixMap(refVolume.gaussians[0],dbVolume.gaussians[1])
       
R1 = np.array([0,0,0])
    #R2 = np.array([1.2,1.2,2.0])
    
atomD = AtomGaussian()
atomD.alpha = 0.836674050
atomD.l = 1
atomD.centre = R1       
       
atomC = AtomGaussian()
atomC.alpha = 0.836674050
atomC.l = 1

  
bender1 = Bender()
bender1.create_experiment(
    name='SH overlap experiment',
    description='Find the maximum pz overlap',
    metrics=[{"metric_name": "trans", "type": "reward"}],
)

bender1.create_algo(
    name='translate function',
    hyperparameters=[
        {
            "name": 'x1',
            "category": "uniform",
            "search_space": {
                "low": 0,
                "high": 1,
            },
        },
                    
        {
            "name": 'x2',
            "category": "uniform",
            "search_space": {
                "low": 0,
                "high": 1,
            },
        },
                    
                    
         {
            "name": 'x3',
            "category": "uniform",
            "search_space": {
                "low": 0,
                "high": 1,
            },
        },            
    ]
)
    
counter = 0

time_list = []
for _ in range(100):
    
    suggestion = bender1.suggest(metric="trans" , optimizer = "parzen_estimator")
    #trans = np.array((suggestion["x1"] , suggestion["x2"], suggestion["x3"]))

    atomC.centre = np.array([suggestion["x1"],suggestion["x2"],suggestion["x3"]])
      
    S_value = SHoverlap(atomC,atomD).real
       
    bender1.create_trial(
            hyperparameters = suggestion,
                results={"trans": S_value},
                weight = 1
                )  
      
    counter += 1
    
    time_list.append([counter, S_value])
    #if S_value.real > 0.1:
    #    print("centre :", atomC.centre, " S_value :", S_value.real)    
    
#%%
x_value = []
y_value = []    

for i in time_list:
    x_value.append(i[0])
    y_value.append(i[1])



plt.plot (x_value, y_value)
plt.grid()
plt.show()    


#%%

