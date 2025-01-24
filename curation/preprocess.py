from copy import deepcopy

import numpy as np
import pandas as pd
from rdkit import Chem

from utils.chemutils import reset_isotopes

MIN_NUM_OF_PRD  = 100

# Count heavy atoms
hatom_counter   = lambda w : Chem.MolFromSmiles(w).GetNumHeavyAtoms()
# Split per fragments
cpd_splitter    = lambda x : pd.Series(x.split('.'))


def CustomCorrelationChecker(df: pd.DataFrame, index_col, thres: float=0.6):
    d = {
        rct : set(df[df['Rep_reaction']==rct][index_col].to_list())
        for rct in set(df['Rep_reaction'])
        }
    val_counts = pd.Series({key : len(val) for key, val in d.items()}).sort_values(ascending=False)
    to_drop = set()
    for keyi in val_counts.index:
        if keyi in to_drop: continue
        for keyj in val_counts.index:
            if keyi == keyj: continue
            vali_u = set(d[keyi])
            valj_u = set(d[keyj])
            min_len =  min(len(vali_u),len(valj_u))
            min_key = keyi if min_len == len(vali_u) else keyj
            corr = len(vali_u.intersection(valj_u)) / min_len
            if corr > thres: 
                to_drop.add(min_key)
    remain = tuple(key for key in d if key not in to_drop)
    df_remain = df[df['Rep_reaction'].isin(remain)].copy()
    return df_remain


def DataPreprocessing(df, index_col, template_col, product_col, reactants_col, AKA_name_col):
    df_rxncenter_extracted = UniqueReactionCenterExtractor(df.copy(),template_col)
    ResetIsotopesSmiles(df_rxncenter_extracted,product_col)
    HeavyAtomsCounterForAllComponents(df_rxncenter_extracted,product_col,reactants_col)
    df_rxn_grouped = GroupByAKAReactionName(df_rxncenter_extracted,template_col,AKA_name_col)
    df_hatoms_checked = PrecursorsHeavyAtomsChecker(df_rxn_grouped, reactants_col)
    df_topn_extracted = TopNrxnExtractor(df_hatoms_checked,index_col)
    df_highcorr_deled = CustomCorrelationChecker(df_topn_extracted,index_col)
    return df_highcorr_deled


def GroupByAKAReactionName(df: pd.DataFrame, template_col: str, AKA_name_col: str):
    reactions = []
    is_first = True
    for tmp,df_rxn in df.groupby(template_col):
        # Name by uniquenized USPTO-registered name
        num_uspto = df_rxn.value_counts(AKA_name_col)
        reaction  = num_uspto.idxmax()
        rxn_seed  = deepcopy(reaction)
        cnt = 0
        while rxn_seed in reactions:
            rxn_seed = f'{reaction}_{cnt}'
            cnt += 1
        reactions.append(rxn_seed)
        df_rxn['Rep_reaction'] = rxn_seed

        df_rxn_integrated = df_rxn.copy() \
            if is_first else pd.concat([df_rxn_integrated, df_rxn])
        is_first = False
    assert df_rxn_integrated.drop_duplicates(template_col).shape[0] == df_rxn_integrated.drop_duplicates('Rep_reaction').shape[0]
    return df_rxn_integrated


def HeavyAtomsCounterForAllComponents(df: pd.DataFrame, product_col: str='Product', reactants_col: str='Precursors'):
    df[f'{product_col}_num_of_hatoms'] = df[product_col].apply(hatom_counter)
    df[[f'{reactants_col}1',f'{reactants_col}2']] = df[reactants_col].apply(cpd_splitter)
    df[f'{reactants_col}1_num_of_hatoms'] = df[f'{reactants_col}1'].apply(hatom_counter)
    df[f'{reactants_col}2_num_of_hatoms'] = df[f'{reactants_col}2'].apply(hatom_counter)


def IsSatisfyingDataNum(df: pd.DataFrame, unique_column: str='template'):
    num_counts = pd.DataFrame(df.value_counts(unique_column))
    return num_counts[num_counts['count']>=MIN_NUM_OF_PRD].reset_index()


def PrecursorsHeavyAtomsChecker(df: pd.DataFrame, reactants_col):
    frs_num_of_hatoms = pd.concat([
        pd.DataFrame(
            df.loc[:,
                [f'{reactants_col}1',f'{reactants_col}1_num_of_hatoms']].to_numpy(),
            columns=[f'{reactants_col}','num_of_hatoms']),
        pd.DataFrame(
            df.loc[:,
                [f'{reactants_col}2',f'{reactants_col}2_num_of_hatoms']].to_numpy(),
            columns=[f'{reactants_col}','num_of_hatoms'])])
    frs_num_of_hatoms.drop_duplicates(f'{reactants_col}',inplace=True)
    frs_num_of_hatoms['num_of_hatoms'] = \
        frs_num_of_hatoms['num_of_hatoms'].astype('int')
    frs_described = frs_num_of_hatoms['num_of_hatoms'].describe()
    frs_iqr = frs_described['75%'] - frs_described['25%']
    frs_max = frs_described['75%'] + (1.5 * frs_iqr)
    frs_min = np.min([0, frs_described['25%'] - (1.5 * frs_iqr)])

    df = df[
        (frs_min<=df['Precursors1_num_of_hatoms']) & 
        (frs_max>=df['Precursors1_num_of_hatoms']) & 
        (frs_min<=df['Precursors2_num_of_hatoms']) & 
        (frs_max>=df['Precursors2_num_of_hatoms'])
            ].copy()
    return df


def ResetIsotopesSmiles(df: pd.DataFrame, product_col: str='Product'):
    df[f'{product_col}_raw'] = df[product_col].apply(reset_isotopes)


def Summarizer(data,index_col):
    columns_for_summary = ['template','represent_reaction',
        'num_of_reactants','num_of_products','Product_num_of_hatoms_mean',
        'Product_num_of_hatoms_std','Fragment1_num_of_hatoms_mean',
        'Fragment1_num_of_hatoms_std',
        'Fragment2_num_of_hatoms_mean','Fragment2_num_of_hatoms_std']
    summary = []

    for reaction in set(data['Rep_reaction']):
        data_ddup = data.drop_duplicates(['Rep_reaction',index_col])
        fr_rxn  = data[data['Rep_reaction'] == reaction].copy()
        prd_rxn = data_ddup[data_ddup['Rep_reaction'] == reaction].copy()
        summary.append([
            fr_rxn['template'].to_list()[0],
            reaction,
            fr_rxn.shape[0],
            prd_rxn.shape[0],
            np.mean(prd_rxn['Product_num_of_hatoms']),
            np.std(prd_rxn['Product_num_of_hatoms']),
            np.mean(fr_rxn['Precursors1_num_of_hatoms']),
            np.std(fr_rxn['Precursors1_num_of_hatoms']),
            np.mean(fr_rxn['Precursors2_num_of_hatoms']),
            np.std(fr_rxn['Precursors2_num_of_hatoms']),
        ])

    summarized_df = pd.DataFrame(summary,columns=columns_for_summary).sort_values(
        'num_of_products',ascending=False).reset_index(drop=True)
    
    return summarized_df


def TopNrxnExtractor(df,index_col):
    df_num = df.drop_duplicates(
        ['Rep_reaction',index_col])['Rep_reaction'].value_counts()
    reactions_over_thres = df_num[df_num >= MIN_NUM_OF_PRD].head(10).index.to_list()
    df_extracted = df[df['Rep_reaction'].isin(reactions_over_thres)].copy()
    return df_extracted


def UniqueReactionCenterExtractor(df: pd.DataFrame, template_col: str='template'):
    num_counts_thres = IsSatisfyingDataNum(df, template_col)
    num_counts_thres['left'] = \
        num_counts_thres[template_col].apply(lambda x : x.split('>>')[0])
    num_counts_group = num_counts_thres.groupby('left')['count'].idxmax()
    idx = num_counts_group.to_list()
    num_dropdups = num_counts_thres.loc[idx]
    num_dropdups.sort_values('count',ascending=False,inplace=True)
    return df[df[template_col].isin(num_dropdups[template_col])].copy()

