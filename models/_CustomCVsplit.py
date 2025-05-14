import numpy as np
import pandas as pd
from rdkit import Chem
import collections as cls
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from copy import deepcopy
from models._kernel_and_mod import funcTanimotoSklearn
from utils.chemutils import MorganbitCalcAsVectors

hatom_counter   = lambda w : Chem.MolFromSmiles(w).GetNumHeavyAtoms()

rng = np.random.default_rng(seed=0)

def _iternotisin(cell, list):
    for i, c in enumerate(cell.split('.')):
        try:
            if c in list[i]:
                return False
        except Exception as e:
            print(e)
            return False
    return True

def _mixturesortedlist(l):
    mixl = []
    i, j = 0, len(l) - 1
    while i <= j:
        if i == j:  # リストの中央に達した場合は要素を1回だけ追加
            mixl.append(l[i])
        else:
            mixl.append(l[i])
            mixl.append(l[j])
        i += 1
        j -= 1
    return mixl

def CustomRandomSplit(df:pd.DataFrame, group_index_col:str, group_subgroup_col:str=None, test_size=0.4, n_iter=100):
    group_idx = sorted(list(set(df[group_index_col])))

    maxdistances_subgr = []
    min_distance_range = float('inf')

    if group_subgroup_col is None:
        print('"group_subgroup_col" is not defined. Return exactly one split with random_seed=0.')
        tr_idx, ts_idx = train_test_split(group_idx,test_size=test_size,random_state=0)
        return {'train' : tr_idx, 'test' : ts_idx}

    for i in range(n_iter):
        tr_idx, ts_idx = train_test_split(group_idx,test_size=test_size,random_state=i)
        tr_ratio_subgr = df.drop_duplicates(group_index_col).groupby(group_subgroup_col)[group_index_col].apply(
            lambda x: x[x.isin(tr_idx)].shape[0] / x.shape[0])
        tr_ratio_whole = df.groupby(group_subgroup_col)[group_index_col].apply(
            lambda x: x[x.isin(tr_idx)].shape[0] / x.shape[0])

        # Check grouped reactions are existing in each of dataset
        while not(sorted(tr_ratio_subgr.index.to_list()) == sorted(tr_ratio_whole.index.to_list())):
            for idx in tr_ratio_whole.index.to_list():
                if not(idx in tr_ratio_subgr.index.to_list()):
                    tr_ratio_subgr[idx] = 0

        mmdist_subgr = abs(tr_ratio_subgr.max() - 1 + test_size) + abs(tr_ratio_subgr.min() - 1 + test_size) \
            if tr_ratio_subgr.max() < 0.95 and tr_ratio_subgr.min() > 0.05 else float('inf')
        maxdistances_subgr.append(mmdist_subgr)
        if mmdist_subgr < min_distance_range:
            min_distance_range = mmdist_subgr
            idx_mindistance  = i
            train_test_index = {
                'train' : tr_idx, 'test' : ts_idx
            }
            train_test_ratio = {
                'subgr' : tr_ratio_subgr, 'whole' : tr_ratio_whole
            }
    assert min_distance_range < float('inf')
    print(f'The minimum distance between max and min is seed {idx_mindistance}.')

    return train_test_index, train_test_ratio

def CustomDissimilarRandomSplit(df:pd.DataFrame, group_index_col:str, group_subgroup_col:str, level:int=1, prd_smiles_col='Product_ECFP', rct_smiles_col='Precursors', sim_thres=0.8, test_size=0.4):
    train_df       = pd.DataFrame(columns=df.columns)
    train_whole_df = pd.DataFrame(columns=df.columns)
    test_df        = pd.DataFrame(columns=df.columns)
    test_whole_df  = pd.DataFrame(columns=df.columns)
    for name, subgroup in df.groupby(group_subgroup_col):
        ttidx = CustomRandomSplit(subgroup,group_index_col,test_size=test_size)
        train = subgroup[subgroup[group_index_col].isin(ttidx['train'])].copy()
        train_whole = df[~df[group_index_col].isin(ttidx['test'])].copy()
        train_whole[group_subgroup_col] = name
        test  = subgroup[subgroup[group_index_col].isin(ttidx['test'])]
        train_bits  = csr_matrix(np.array(MorganbitCalcAsVectors(train[prd_smiles_col],n_jobs=-1))).astype(bool)
        train_whole_bits  = csr_matrix(np.array(MorganbitCalcAsVectors(train_whole[prd_smiles_col],n_jobs=-1))).astype(bool)
        test_bits   = csr_matrix(np.array(MorganbitCalcAsVectors(test[prd_smiles_col],n_jobs=-1))).astype(bool)
        sim_array   = funcTanimotoSklearn(test_bits,train_bits)
        sim_array_whole   = funcTanimotoSklearn(test_bits,train_whole_bits)
        max_sim_ts  = np.max(sim_array,axis=1)
        max_sim_ts_whole  = np.max(sim_array_whole,axis=1)
        is_over_thres     = (max_sim_ts < sim_thres)
        is_over_thres_whole         = (max_sim_ts_whole < sim_thres)
        test_dissim       = test[pd.Series(is_over_thres,index=test.index)].copy()
        test_whole_dissim = test[pd.Series(is_over_thres_whole,index=test.index)].copy()
        train_df       = pd.concat([train_df, train])
        train_whole_df = pd.concat([train_whole_df, train_whole])
        if level <= 1:
            test_df        = pd.concat([test_df, test_dissim])
            test_whole_df  = pd.concat([test_whole_df, test_whole_dissim])
        else:
            rct_pool       = [list(x) for x in zip(*[rct.split('.') for rct in train[rct_smiles_col]])]
            rct_whole_pool = [list(x) for x in zip(*[rct.split('.') for rct in train_whole[rct_smiles_col]])]
            test_dissim['is_unique'] = test_dissim[rct_smiles_col].apply(_iternotisin,args=(rct_pool,))
            test_whole_dissim['is_unique'] = test_whole_dissim[rct_smiles_col].apply(_iternotisin,args=(rct_whole_pool,))
            test_df        = pd.concat([test_df, test_dissim[test_dissim['is_unique']]])
            test_whole_df  = pd.concat([test_whole_df, test_whole_dissim[test_whole_dissim['is_unique']]])
    return train_df, train_whole_df, test_df, test_whole_df

def CustomFragmentSpaceSplitbyFreq(df:pd.DataFrame, group_index_col:str, rct_smiles_col='smiles', test_ratio=0.4, group_subgroup_col:str=None, tol=0.05, max_iter=1000):
    assert(test_ratio>0 and test_ratio<1)
    train_df       = pd.DataFrame(columns=df.columns)
    train_whole_df = pd.DataFrame(columns=df.columns)
    test_df        = pd.DataFrame(columns=df.columns)
    test_whole_df  = pd.DataFrame(columns=df.columns)
    df_subgroup = df.groupby(group_subgroup_col) if group_subgroup_col is not None else (('_',df,),)
    plot_corr = dict()
    for name, subgroup in df_subgroup:
        frs_pair   = [smi.split('.') for smi in subgroup[rct_smiles_col]]
        gr_indices = subgroup[group_index_col].copy()
        gr_dict    = {gr : i for i, gr in enumerate(sorted(set(gr_indices)))}
        frs_pair_T = [list(x) for x in zip(*frs_pair)]
        fr1_mcn, _ = zip(*cls.Counter(frs_pair_T[0]).most_common())
        fr2_mcn, _ = zip(*cls.Counter(frs_pair_T[1]).most_common())
        fr1_dict = {fr1 : i for i, fr1 in enumerate(_mixturesortedlist(fr1_mcn))}
        fr2_dict = {fr2 : i for i, fr2 in enumerate(fr2_mcn[::-1])}
        frs_coor = np.zeros((len(frs_pair), 3))
        for i, (gr, pair) in enumerate(zip(gr_indices,frs_pair)):
            frs_coor[i, 0] = fr1_dict[pair[0]]
            frs_coor[i, 1] = fr2_dict[pair[1]]
            frs_coor[i, 2] = gr_dict[gr]
        frs_max = np.max(frs_coor, axis=0)[:-1]
        ts_ratio =test_ratio
        best_ratio_diff = float('inf')
        for i, _ in enumerate(range(max_iter)):
            tr_ratio = 1 - ts_ratio
            ratio_to_split = np.sqrt(tr_ratio) / (np.sqrt(tr_ratio) + np.sqrt(ts_ratio))
            split_point = frs_max * ratio_to_split
            train_indices = list()
            gr_cache = set()
            test_indices  = list()
            for j, pair in enumerate(frs_coor):
                if all(pair[:-1] <= split_point):
                    train_indices.append(j)
                    gr_cache.add(pair[-1])
            for j, pair in enumerate(frs_coor):
                if all(pair[:-1] > split_point) and (pair[-1] not in gr_cache):
                    test_indices.append(j)
            ratio_ts = len(test_indices) / (len(train_indices) + len(test_indices))
            if (ratio_ts - test_ratio) < best_ratio_diff:
                best_ratio = (tr_ratio, ts_ratio)
                best_ratio_diff = (ratio_ts - test_ratio)
                best_train_indices = deepcopy(train_indices)
                best_test_indices  = deepcopy(test_indices)
            if (ratio_ts - test_ratio) < tol: break
            ts_ratio = ts_ratio - 0.01 if (ts_ratio - 0.01) > 0 else 0.99
        if i==(max_iter-1): 
            print('Max iter warning !')
            tr_ratio, ts_ratio = best_ratio
            ratio_to_split = np.sqrt(tr_ratio) / (np.sqrt(tr_ratio) + np.sqrt(test_ratio))
            split_point = frs_max * ratio_to_split
            train_indices = best_train_indices
            test_indices  = best_test_indices
        tr_df = subgroup.iloc[train_indices]
        ts_df = subgroup.iloc[test_indices]
        tr_wh_df = df[~df[group_index_col].isin(set(ts_df[group_index_col]))]
        tr_wh_df['Rep_reaction'] = name
        ts_wh_df = ts_df.copy()
        ts_wh_df['Rep_reaction'] = name
        train_df = pd.concat([train_df, tr_df])
        test_df  = pd.concat([test_df, ts_df])
        assert(len(set(tr_df[group_index_col]).intersection(set(ts_df[group_index_col])))==0)
        assert(len(set([rcts.split('.')[0] for rcts in tr_df[rct_smiles_col]]).intersection(set([rcts.split('.')[0] for rcts in ts_df[rct_smiles_col]])))==0)
        assert(len(set([rcts.split('.')[1] for rcts in tr_df[rct_smiles_col]]).intersection(set([rcts.split('.')[1] for rcts in ts_df[rct_smiles_col]])))==0)
        train_whole_df = pd.concat([train_whole_df, tr_wh_df])
        test_whole_df  = pd.concat([test_whole_df, ts_wh_df])
        plot_corr[name] = (fr1_dict, fr2_dict, frs_coor)
    return train_df, train_whole_df, test_df, test_whole_df, plot_corr
