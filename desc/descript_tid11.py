'''Descriptor生成用コード
    RDKitを用いてtid=11（分子数1235）のフラグメントのDescriptorsを算出
'''

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors

path = 'recap_tid11_grouping_amide.tsv'

tid_11 = pd.read_table(path, index_col=0, header=0)
#print(tid_11)

smiles_list_fragment1 = np.array(tid_11.copy().loc[:, 'Fragment1'])
smiles_list_fragment2 = np.array(tid_11.copy().loc[:, 'Fragment2'])
mol_list_fragment1 = []
mol_list_fragment2 = []

for idx, id in enumerate(tid_11['ddc_id']):
    pro = tid_11.iloc[idx, 1]
    val = tid_11.iloc[idx, 2]
    mol_fragment1 = Chem.MolFromSmiles(smiles_list_fragment1[idx])
    mol_list_fragment1.append([id, pro, val, smiles_list_fragment1[idx], mol_fragment1])
    mol_fragment2 = Chem.MolFromSmiles(smiles_list_fragment2[idx])
    mol_list_fragment2.append([id, pro, val, smiles_list_fragment2[idx], mol_fragment2])

fragment1 = pd.DataFrame(mol_list_fragment1,index=None,columns=['ddc_id','Product', 'pot.(log,Ki)', 'Fragment1','Fragment1_mol'])
fragment2 = pd.DataFrame(mol_list_fragment2,index=None,columns=['ddc_id','Product', 'pot.(log,Ki)', 'Fragment2','Fragment2_mol'])
# tid_11_smiles = pd.DataFrame(tid_11.copy().iloc[:, 4],) # 4 means washed_openeye_smiles
# tid_11_smiles['mol'] = mol_list
# mol_list = [Chem.MolFromSmiles('CCN(CCC)CCCC')]
# index = ['test']

#tid_11_smiles

for i,j in Descriptors.descList:
    fragment1[i] = fragment1['Fragment1_mol'].map(j)
    fragment2[i] = fragment2['Fragment2_mol'].map(j)

fragment1.to_csv('recap_tid11_grouping_amide_descriptors_fragment1.tsv',sep='\t')
fragment2.to_csv('recap_tid11_grouping_amide_descriptors_fragment2.tsv',sep='\t')