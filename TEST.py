# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:11:32 2021

@author: 86153
"""
import numpy as np
from rdkit import Chem
from AtomGaussian import AtomGaussian
from GaussianVolume import GaussianVolume
from ShapeAlignment import ShapeAlignment
from SolutionInfo import SolutionInfo, updateSolutionInfo


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