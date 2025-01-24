'''Code for calcurating descriptors
    Calcurate descriptors using RDKit
'''

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

if __name__ == '__main__':
    path = 'recap_tid11_grouping_amide.tsv'

def return_mols(smiles_list):
    mol_list_rdkit = []
    mol_list_ecfp = []
    for smiles in smiles_list:
        mol_rdkit = Chem.MolFromSmiles(smiles)
        mol_list_rdkit.append([smiles, mol_rdkit])
        mol_ecfp = Chem.MolFromSmiles(smiles)
        mol_list_ecfp.append([smiles, mol_ecfp])
    return mol_list_rdkit, mol_list_ecfp

def calc_ecfp(mol_list, fr=None):
    ecfp_list = []
    ecfp_bitinfo = []
    if fr != None:
        ecfp_col = [f"fr{fr}_ecfp_bit_{i}" for i in range(1024*8)]
        ecfp_col[0:0] = [f'Fragment{fr}_ECFP']
    else:
        ecfp_col = [f"ecfp_bit_{i}" for i in range(1024*8)]
        ecfp_col[0:0] = [f'Product_ECFP']
    for list in mol_list:
        ecfp_list_tmp = [list[0]]
        ecfp_bitinfo_tmp = {}
        ecfp = AllChem.GetMorganFingerprintAsBitVect(list[1], 2, 1024*8, 
            useChirality=True, bitInfo=ecfp_bitinfo_tmp)
        for bit in ecfp:
            ecfp_list_tmp.extend([bit])
        ecfp_list.append(ecfp_list_tmp)
        ecfp_bitinfo.append([list[0],ecfp_bitinfo_tmp])
    return ecfp_list, ecfp_col, ecfp_bitinfo

def calc_descs(mol_list,fr=None):
    desc_list = []
    desc_col = [desc[0] for desc in Descriptors.descList]
    if fr!=None:
        desc_col[0:0] = [f'Fragment{fr}_rdkit']
    else:
        desc_col[0:0] = [f'Product_rdkit']
    for list in mol_list:
        mol = list[1]
        desc_list_tmp = [list[0]]
        for desc in Descriptors.descList:
            desc_list_tmp.append(desc[1](mol))
        desc_list.append(desc_list_tmp)
    return desc_list, desc_col

if __name__ == '__main__':
    tid = pd.read_table(path, index_col=0, header=0)

    smiles_list_fr1 = np.array(tid.copy().loc[:, 'Fragment1'])
    smiles_list_fr2 = np.array(tid.copy().loc[:, 'Fragment2'])

    mol_list_rdkit_fr1, mol_list_ecfp_fr1 = return_mols(smiles_list_fr1)
    mol_list_rdkit_fr2, mol_list_ecfp_fr2 = return_mols(smiles_list_fr2)

    rdkit_fr1 = pd.concat([tid.loc[:,:'pot.(log,Ki)'],pd.DataFrame(
        mol_list_rdkit_fr1, index=tid.index, columns=['Fragment1_rdkit','Fragment1_rdkit_mol'])],axis=1)
    rdkit_fr2 = pd.concat([tid.loc[:,:'pot.(log,Ki)'],pd.DataFrame(
        mol_list_rdkit_fr2, index=tid.index, columns=['Fragment2_rdkit','Fragment2_rdkit_mol'])],axis=1)

    for i,j in Descriptors.descList:
        rdkit_fr1[i] = rdkit_fr1['Fragment1_rdkit_mol'].map(j)
        rdkit_fr2[i] = rdkit_fr2['Fragment2_rdkit_mol'].map(j)

    ecfp_fr1, ecfp_fr1_col = calc_ecfp(mol_list_ecfp_fr1, 1)
    ecfp_fr2, ecfp_fr2_col = calc_ecfp(mol_list_ecfp_fr2, 2)

    fr1 = pd.concat([rdkit_fr1, pd.DataFrame(ecfp_fr1,index=tid.index,columns=ecfp_fr1_col)],axis=1)
    fr2 = pd.concat([rdkit_fr2, pd.DataFrame(ecfp_fr2,index=tid.index,columns=ecfp_fr2_col)],axis=1)

    fr1.dropna(axis=1)
    fr2.dropna(axis=1)

    fr1.to_csv('recap_tid11_grouping_amide_descriptors_ecfp_fragment1_fragmented.tsv',sep='\t')
    fr2.to_csv('recap_tid11_grouping_amide_descriptors_ecfp_fragment2_fragmented.tsv',sep='\t')