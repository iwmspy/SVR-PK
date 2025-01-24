'''Descriptor生成用コード
    RDKitを用いてtid=11（分子数1235）のフラグメントのDescriptorsを算出
'''

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

if __name__ == '__main__':
    path = 'recap_tid11_grouping_amide.tsv'

def descript(smiles_list):
    mol_list_rdkit = []
    mol_list_ecfp = []
    for i, smiles in enumerate(smiles_list):
        print(f'combert smiles to mol No.{i}')
        mol_rdkit = Chem.MolFromSmiles(smiles)
        mol_list_rdkit.append([smiles, mol_rdkit])
        # mol_rdkit = Chem.MolFromSmiles(smiles.replace('*',''))
        # mol_list_rdkit.append([smiles.replace('*',''), mol_rdkit])
        # mol_ecfp = Chem.MolFromSmiles(smiles.replace('*','[3H]'))
        mol_ecfp = Chem.MolFromSmiles(smiles.replace('*',''))
        # mol_list_ecfp.append([smiles.replace('*','[3H]'), mol_ecfp])
        mol_list_ecfp.append([smiles.replace('*',''), mol_ecfp])
    return mol_list_rdkit, mol_list_ecfp

def ecfp(mol_list, fr):
    ecfp_list = []
    ecfp_col = [f"fr{fr}_ecfp_bit_{i}" for i in range(1024*8)]
    ecfp_col[0:0] = [f'Fragment{fr}_ECFP_smiles',f'Fragment{fr}_ECFP_mol']
    for list in mol_list:
        ecfp_list_tmp = []
        ecfp = AllChem.GetMorganFingerprintAsBitVect(list[1], 2, 1024*8)
        ecfp_list_tmp.extend([list[0], list[1]])
        for bit in ecfp:
            ecfp_list_tmp.extend([bit])
        ecfp_list.append(ecfp_list_tmp)
    return ecfp_list, ecfp_col

if __name__ == '__main__':
    tid_11 = pd.read_table(path, index_col=0, header=0)
    #print(tid_11)

    smiles_list_fr1 = np.array(tid_11.copy().loc[:, 'Fragment1'])
    smiles_list_fr2 = np.array(tid_11.copy().loc[:, 'Fragment2'])
    # mol_list = []
    # mol_list_fr2 = []

    mol_list_rdkit_fr1, mol_list_ecfp_fr1 = descript(smiles_list_fr1)
    mol_list_rdkit_fr2, mol_list_ecfp_fr2 = descript(smiles_list_fr2)

    rdkit_fr1 = pd.concat([tid_11.loc[:,:'pot.(log,Ki)'],pd.DataFrame(
        mol_list_rdkit_fr1, index=tid_11.index, columns=['Fragment1_rdkit','Fragment1_rdkit_mol'])],axis=1)
    rdkit_fr2 = pd.concat([tid_11.loc[:,:'pot.(log,Ki)'],pd.DataFrame(
        mol_list_rdkit_fr2, index=tid_11.index, columns=['Fragment2_rdkit','Fragment2_rdkit_mol'])],axis=1)

    # for idx, id in enumerate(tid_11['ddc_id']):
    #     pro = tid_11.iloc[idx, 1]
    #     val = tid_11.iloc[idx, 2]
    #     mol_fragment1 = Chem.MolFromSmiles(smiles_list_fr1[idx])
    #     mol_list.append([id, pro, val, smiles_list_fr1[idx], mol_fragment1])
    #     mol_fragment2 = Chem.MolFromSmiles(smiles_list_fr2[idx])
    #     mol_list_fr2.append([id, pro, val, smiles_list_fr2[idx], mol_fragment2])

    # rdkit_fr1 = pd.DataFrame(mol_list,index=None,columns=['ddc_id','Product', 'pot.(log,Ki)', 'Fragment1','Fragment1_mol'])
    # rdkit_fr2 = pd.DataFrame(mol_list_fr2,index=None,columns=['ddc_id','Product', 'pot.(log,Ki)', 'Fragment2','Fragment2_mol'])
    # tid_11_smiles = pd.DataFrame(tid_11.copy().iloc[:, 4],) # 4 means washed_openeye_smiles
    # tid_11_smiles['mol'] = mol_list
    # mol_list = [Chem.MolFromSmiles('CCN(CCC)CCCC')]
    # index = ['test']

    #tid_11_smiles

    for i,j in Descriptors.descList:
        rdkit_fr1[i] = rdkit_fr1['Fragment1_rdkit_mol'].map(j)
        rdkit_fr2[i] = rdkit_fr2['Fragment2_rdkit_mol'].map(j)

    ecfp_fr1, ecfp_fr1_col = ecfp(mol_list_ecfp_fr1, 1)
    ecfp_fr2, ecfp_fr2_col = ecfp(mol_list_ecfp_fr2, 2)

    fr1 = pd.concat([rdkit_fr1, pd.DataFrame(ecfp_fr1,index=tid_11.index,columns=ecfp_fr1_col)],axis=1)
    fr2 = pd.concat([rdkit_fr2, pd.DataFrame(ecfp_fr2,index=tid_11.index,columns=ecfp_fr2_col)],axis=1)

    fr1.to_csv('recap_tid11_grouping_amide_descriptors_ecfp_fragment1_removed.tsv',sep='\t')
    fr2.to_csv('recap_tid11_grouping_amide_descriptors_ecfp_fragment2_removed.tsv',sep='\t')