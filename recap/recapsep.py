'''Code for RECAP
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

# These are the definitions that will be applied to fragment molecules:
reactionDefs = {
  # "urea" : "[#7;+0;D2,D3:1]!@C(!@=O)!@[#7;+0;D2,D3:2]>>*[#7:1].[#7:2]*",  # Ref
  "amide" : "[C;!$(C([#7])[#7]):1](=!@[O:2])!@[#7;+0;!D1:3]>>*[C:1]=[O:2].*[#7:3]", # Ref
  "ester" : "[C:1](=!@[O:2])!@[O;+0:3]>>*[C:1]=[O:2].[O:3]*",  # Ref
  # "amines" : "[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*:1])-!@[*:2]>>*[*:1].[*:2]*",  # Ref
  "cyclic amines" : "[#7;R;D3;+0:1]-!@[*:2]>>*[#7:1].[*:2]*", # RDKit
  # "ether" : "[#6:1]-!@[O;+0]-!@[#6:2]>>[#6:1]*.*[#6:2]",  # Ref
  # "olefin" : "[C:1]=!@[C:2]>>[C:1]*.*[C:2]", # Ref
  "aromatic nitrogen - aliphatic carbon" : "[n;+0:1]-!@[C;!$(C@*);!$(C@@*):2]>>[n:1]*.[C:2]*", # Ref
  "lactam nitrogen - aliphatic carbon" : "[O:3]=[C:4]-@[N;+0:1]-!@[C;!$(C@*);!$(C@@*):2]>>[O:3]=[C:4]-[N:1]*.[C:2]*", # Ref
  "aromatic carbon - aromatic carbon" : "[c:1]-!@[c:2]>>[c:1]*.*[c:2]", # Ref
  "aromatic nitrogen - aromatic carbon" : "[n;+0:1]-!@[c:2]>>[n:1]*.*[c:2]", # RDKit
  "sulphonamide" : "[#7;+0;D2,D3:1]-!@[S:2](=[O:3])=[O:4]>>[#7:1]*.*[S:2](=[O:3])=[O:4]", # Ref
}

reactions = tuple([Reactions.ReactionFromSmarts(x) for x in reactionDefs.values()])

reactionDefs_canonical = reactionDefs.copy()
for key in reactionDefs_canonical.keys():
  reactionDefs_canonical[key] = Reactions.ReactionToSmarts(
    Reactions.ReactionFromSmarts(reactionDefs_canonical[key]))

def sep(smi):
  """ returns the recap decomposition for a molecule """
  mol = Chem.MolFromSmiles(smi)
  frag_list = []

  t1 = time.time()

  for reaction in reactions:
    ps = reaction.RunReactants((mol, ))
    for key, reactionDef in reactionDefs_canonical.items():
      if Reactions.ReactionToSmarts(reaction)==reactionDef:
        reactionType=key
        break
    for reactant in ps:
      tmp = [Chem.MolToSmiles(mol), 
             f'{Chem.MolToSmiles(reactant[0])}.{Chem.MolToSmiles(reactant[1])}', 
             Reactions.ReactionToSmarts(reaction), reactionType]
      if not tmp in frag_list:
        frag_list.append(tmp)
  t2 = time.time()
  run_time = t2 - t1
  
  print(f"took {run_time} seconds to fragmentation")

  return frag_list

def analysis_of_recap(df:pd.DataFrame,name='recap_summary.txt'):
    f = open(name,'w')
    f.write(f'==={name.split(".")[0]}=== \n \n')

    f.write('<Number of class appearances>\n')
    f.write(df['class'].value_counts().to_string())

    f.close()

if __name__ == '__main__':
  #for example
  smiles = 'N=C(N)c1cccc(C[C@H](NS(=O)(=O)c2ccc3ccccc3c2)C(=O)N2CCC(C3CCN(C(=N)N)CC3)CC2)c1'

  recap = sep(smiles)

  print(recap)
