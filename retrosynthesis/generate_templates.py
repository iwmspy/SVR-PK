'''Code for Retrosynthesis
    Fragmentation using RECAP 
    (RECAP : REtrosynthetic Combinatorial Analysis Procedure)
    Ref : https://pubs.acs.org/doi/10.1021/ci970429i
    Rules : 11 rules from ref and 2 rules from RDKit
    
    !!!Note!!!
    Due to split position issues and stereo- or geometric-isomers, 
    some fragmentation methods are deleted or modified.
    Deleteds are indicated as comment.
'''

import pandas as pd
import numpy as np
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, Recap
import time
import sys
import weakref
from rdkit.Chem import rdChemReactions as Reactions

if __name__ == '__main__':
  import os
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

class generate_templates:
  def __init__(self, df:pd.DataFrame, rxn_col:str, class_col:str=None):
    self.df = df
    self.rxns = df.loc[:,rxn_col].to_list()
    if class_col!=None:
      self.class_col = df.loc[:,class_col].to_list

  def smiles_to_mols(self,smiles:str):
    smiles_list = smiles.split('.')
    mol_list = []
    for smiles_ in smiles_list:
      mol = Chem.MolFromSmiles(smiles_)
      mol_list.append(mol)
    
    return mol_list
  
  # def mols_from_smarts(self,mols):
  #   for mol in mols:
      
  #   return

  def generate_templates(self):
    # canonicalize
    for rxn in self.rxns:
      try:
        rxn_list = rxn.split('>>')
        if len(rxn_list)!=2:
          print(f'{rxn} has invalid number of reactions (number of reactions:{len(rxn_list)})')
          continue
        reac = rxn_list[0]
        prod = rxn_list[1]
        reac_list = self.smiles_to_mols(reac)
        prod_list = self.smiles_to_mols(prod)

      except:
        print(f'Cannot generate template with {rxn}')

if __name__ == '__main__':
  #for example
  id = 'ddc_306194'
  smiles = 'N=C(N)c1cccc(C[C@H](NS(=O)(=O)c2ccc3ccccc3c2)C(=O)N2CCC(C3CCN(C(=N)N)CC3)CC2)c1'
  mol = Chem.MolFromSmiles(smiles)
  obv = 5.74

  # recap = sep(mol, id, obv)

  # print(recap)
