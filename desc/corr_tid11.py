'''Descriptor生成用コード
    RDKitを用いてtid=11（分子数1235）のフラグメントのDescriptorsを算出
'''

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors
import re
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    path_1 = 'recap_tid11_grouping_amide_descriptors_ecfp_fragment1_fragmented.tsv'
    path_2 = 'recap_tid11_grouping_amide_descriptors_ecfp_fragment2_fragmented.tsv'
    path_all = 'recap_tid11_grouping_amide.tsv'

def onlynum(s):
    for index in range(s.shape[0]):
        for column in range(s.shape[1]):
            if not(re.compile('[0-9.-]+').fullmatch(str(s.iloc[index, column]))):
                s.iloc[index, column] = np.nan
    s_drop = s.dropna(axis=1)
    return s_drop

def valuedifference(column):
    sum = column.shape[0]
    mode = column.value_counts().iat[0]
    calc = mode / sum
    if calc < 0.95:
        return True
    else: 
        return False

def chk(column):
    x = 0
    for l in column:
        if abs(l) >= 0.6 and abs(l) < 1:
            x = x + abs(l)
    return x

def correlation(data, file_name='correration.png', mapping='hsv', mask=None, fig_size=[70,60]):
    # data = data.iloc[:, data.columns.get_loc('Fragment2_mol')+1:]
    prep_1 = onlynum(data)
    prep_2 = pd.DataFrame(index=data.index)
    for i in range(prep_1.shape[1]):
     if valuedifference(prep_1.iloc[:, i]):
          concat = pd.DataFrame(prep_1.iloc[:, i])
          prep_2 = pd.concat([prep_2, concat], axis=1)
    corr = prep_2.copy().corr()
    plt.figure(figsize=(fig_size[0],fig_size[1]))
    if mask=='high':
        corr_h = (corr.abs()>=0.6)
        sns.heatmap(corr,cmap=mapping,annot=True,mask=corr_h)
    elif mask=='low':
        corr_l = (corr.abs()<=0.6)
        sns.heatmap(corr,cmap=mapping,annot=True,mask=corr_l)
    elif mask==None:
        sns.heatmap(corr,cmap=mapping,annot=True)
    else:
        Exception("mask error : opt 'high', 'low', None")
    sns.heatmap(corr,cmap='hsv',annot=True,mask=corr_l)
    plt.tight_layout()
    plt.savefig(file_name)

if __name__ == '__main__':
    tid_11_fragment1 = pd.read_table(path_1, index_col=0, header=0)
    tid_11_fragment1_desc = tid_11_fragment1.copy().iloc[
        :,tid_11_fragment1.columns.get_loc('Fragment1_rdkit_mol')+1:tid_11_fragment1.columns.get_loc('Fragment1_ECFP_smiles')-1].add_prefix('fr01_')
    tid_11_fragment2 = pd.read_table(path_2, index_col=0, header=0)
    tid_11_fragment2_desc = tid_11_fragment2.copy().iloc[
        :,tid_11_fragment2.columns.get_loc('Fragment2_rdkit_mol')+1:tid_11_fragment2.columns.get_loc('Fragment2_ECFP_smiles')-1].add_prefix('fr02_')
    #print(tid_11)

    tid_11_concat_stable = tid_11_fragment1.copy().loc[:, :'Fragment1_rdkit_mol']
    tid_11_concat_stable = pd.concat([tid_11_concat_stable, tid_11_fragment2.loc[
        :,'Fragment2_rdkit':'Fragment2_rdkit_mol']],axis=1)

    tid_11_fragment1_desc_raw, tid_11_fragment1_desc_select = correlation(tid_11_concat_stable, tid_11_fragment1_desc)
    tid_11_fragment2_desc_raw, tid_11_fragment2_desc_select = correlation(tid_11_concat_stable, tid_11_fragment2_desc)

    tid_11_concat_desc_raw = pd.concat([tid_11_fragment1_desc_raw, tid_11_fragment2_desc_raw.iloc[
        :,tid_11_fragment2_desc_raw.columns.get_loc('Fragment2_rdkit_mol')+1 :]], axis=1)
    tid_11_concat_desc_selected = pd.concat([tid_11_fragment1_desc_select, tid_11_fragment2_desc_select.iloc[
        :,tid_11_fragment2_desc_select.columns.get_loc('Fragment2_rdkit_mol')+1 :]], axis=1)

    # tid_11_concat_desc_raw, tid_11_concat_desc_selected = preprocess(tid_11_concat_stable, tid_11_concat_desc)
    # tid_11_smiles = pd.DataFrame(tid_11.copy().iloc[:, 4],) # 4 means washed_openeye_smiles
    # tid_11_smiles['mol'] = mol_list
    # mol_list = [Chem.MolFromSmiles('CCN(CCC)CCCC')]
    # index = ['test']

    #tid_11_smiles

    tid_11_concat_desc_raw.to_csv('recap_tid11_grouping_amide_descriptors_average_raw_divided_v2.tsv',sep='\t')
    tid_11_concat_desc_selected.to_csv('recap_tid11_grouping_amide_descriptors_average_selected_divided_v2.tsv',sep='\t')