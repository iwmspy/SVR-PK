from collections import Counter

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from tempfile import TemporaryDirectory
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem.Draw import MolToImage, DrawMorganBit
import seaborn as sns

from retrosynthesis import retrosep
from utils.utility import tsv_merge
from utils.chemutils import MurckoScaffoldSmilesListFromSmilesList, is_valid_molecule
from utils.SA_Score.sascorer import readFragmentScores, calculateScore

chunk = 100000

hist_seed_obj = lambda ax, x, y: ax.hist(x, histtype='step', bins=25, label=y)
hist_seed_hat = lambda ax, x, y: ax.hist(x, bins=25, label=y, alpha=0.5)

descs = {name: method for name, method in Descriptors.descList}
descs['SAscore'] = calculateScore
descs_to_use = ['MolWt', 'HeavyAtomCount', 'NumHDonors', 'NumHAcceptors', 'RingCount', 'TPSA', 'MolLogP','SAscore']

def GridGenerator(unique: set):
    num_to_gen = len(unique)
    vertical   = int(np.floor(np.sqrt(num_to_gen)))
    horizontal = int(np.ceil(num_to_gen / vertical))
    return (vertical, horizontal)

def multicolumnsgenerator(df: pd.DataFrame, *args):
    arrays  = [[x for _ in range(df.shape[1])] for x in args]
    arrays.append(df.columns.to_list())
    columns = pd.MultiIndex.from_arrays(arrays)
    data    = df.to_numpy()
    index   = df.index
    return pd.DataFrame(data, index=index, columns=columns)

def boxplotter(x, y, data, hue, ax, xlabel=False, ylabel=False):
    sns.boxplot(x=x, y=y, data=data, hue=hue, ax=ax)

def boxplotterwithstripplot(x, y, data, hue, ax, xlabel=False, ylabel=False):
    sns.boxplot(x=x, y=y, data=data, hue=hue, ax=ax, showfliers=False)
    sns.stripplot(x=x, y=y, data=data, hue=hue, ax=ax, dodge=True, marker='o',edgecolor="black",color='black',size=5.0,linewidth=1,legend=False)

def vioplotter(x, y, data, hue, ax, xlabel=False, ylabel=False):
    sns.violinplot(x=x, y=y, data=data, hue=hue, ax=ax)

def ValidityAndSuggestedRouteExaminator(ipath,opath,input_smi_col='SMILES',prec_smi_col='Precursors',ext_data_pathes=None,input_index_col=None,ext_index_col=None,ext_smi_col=None,ext_index_sep=None):
    ext_datas = {}
    if ext_data_pathes is not None:
        for i, ext_data_path in enumerate(ext_data_pathes):
            ext_datas[i] = {}
            if ext_index_col is None:
                ext_index_col = 0
            ext_df = pd.read_table(ext_data_path,index_col=ext_index_col,header=0,chunksize=chunk)
            for ext_df_ in ext_df:
                for idx, row in ext_df_.iterrows():
                    if ext_index_col!=0:
                        ext_datas[i][row[ext_smi_col]] = idx
                    else:
                        ext_datas[i][row[ext_smi_col]] = row[ext_smi_col]
    with TemporaryDirectory() as tmpdir:
        shape = 0
        if input_index_col is None:
            input_index_col = 0
        df_chunked  = pd.read_table(ipath,index_col=input_index_col,header=0,chunksize=chunk)
        df_syn_list = []
        for i, sc_preds in enumerate(df_chunked):
            is_top_precursors = []
            for idx, row in sc_preds.iterrows():
                try:
                    assert is_valid_molecule(row[input_smi_col])
                    out = retrosep.do_one(row[input_smi_col],rc_cumurate=False)
                except:
                    out = []
                if len(out) == 0:
                    is_top_precursors.append(False)
                    continue
                out_smi =[o[1] for o in out]
                if ext_data_pathes is None:
                    is_top_precursors.append(row[prec_smi_col] in out_smi)
                else:
                    idx_smi = []
                    for osmi in out_smi:
                        idx_smi_tmp = []
                        for k, smi in enumerate(osmi.split('.')):
                            if ext_index_col != 0:
                                if smi in ext_datas[k]:
                                    idx_smi_tmp.append(str(ext_datas[k][smi]))
                                else:
                                    idx_smi_tmp.append('NONE')
                            else:
                                idx_smi_tmp.append(smi)
                        idx_smi_str = ext_index_sep.join(idx_smi_tmp) if ext_index_sep is not None else '.'.join(idx_smi_tmp)
                        idx_smi.append(idx_smi_str)
                    if ext_index_col != 0:
                        is_top_precursors.append(idx in idx_smi)
                    else:
                        is_top_precursors.append(row[prec_smi_col] in idx_smi)
            sc_preds['is_top_precursors'] = is_top_precursors
            sc_preds_syn = sc_preds[sc_preds['is_top_precursors']].copy()
            if not sc_preds_syn.empty:
                shape += sc_preds_syn.shape[0]
                sc_preds_syn.to_csv(f'{tmpdir}/{i}.tsv',sep='\t')
                df_syn_list.append(f'{tmpdir}/{i}.tsv')
        if len(df_syn_list):
            tsv_merge(df_syn_list,opath)
            return opath, shape
    return None, None


def ScaffoldExtractor(ipath,odir,input_smi_col='SMILES'):
    scaf_idx = {}
    scaf_pts = {}
    scaf_whl = []
    scaf_cnt = 0
    with TemporaryDirectory() as tmpdir:
        df_chunked  = pd.read_table(ipath,index_col=0,header=0,chunksize=chunk)
        for i, df in enumerate(df_chunked):
            scafs = MurckoScaffoldSmilesListFromSmilesList(df[input_smi_col].to_numpy().ravel()).ravel()
            for scaf in scafs:
                scaf_whl.append(scaf)
                if scaf not in scaf_idx:
                    scaf_idx[scaf] = scaf_cnt
                    scaf_pts[scaf] = []
                    scaf_cnt += 1
            df['scaf'] = scafs
            for scaf, idx in scaf_idx.items():
                df_scaf = df[df['scaf']==scaf].copy()
                if not df_scaf.empty:
                    df_scaf.to_csv(f'{tmpdir}/{idx}_{i}.tsv',sep='\t')
                    scaf_pts[scaf] = scaf_pts[scaf] + [f'{tmpdir}/{idx}_{i}.tsv']
        for scaf, paths in scaf_pts.items():
            tsv_merge(paths,f'{odir}/{os.path.basename(ipath).rsplit(".",1)[0]}_scaf_{scaf_idx[scaf]}.tsv')
    scaf_counter    = Counter(scaf_whl)
    scaf_counter_df = pd.DataFrame(scaf_counter.most_common(),columns=['scaf','appearance'])
    scaf_counter_df['index_to_scaf'] = scaf_counter_df['scaf'].map(scaf_idx)
    return scaf_counter_df


def sascorecalculator_mod(smiles,n_jobs=-1):
    assert n_jobs != 0
    readFragmentScores()
    def _scorer(smis):
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        return np.array([calculateScore(m) for m in mols])
    _j = n_jobs if n_jobs > 0 else os.cpu_count() + (n_jobs+1)
    smis_split = np.array_split(np.array(smiles),_j)
    ret = Parallel(n_jobs=_j,backend='threading')(
        [delayed(_scorer)(smis) for smis in smis_split]
    )
    return np.concatenate(ret)


def SAScoreCalculator(ipath,opath,smi_col='SMILES',n_jobs=-1):
    scores_whole = []
    with TemporaryDirectory() as tmpdir:
        paths = []
        df_chunked  = pd.read_table(ipath,index_col=0,header=0,chunksize=chunk)
        for i, df in enumerate(df_chunked):
            sascores = sascorecalculator_mod(df[smi_col].to_numpy().ravel()).ravel()
            df['sa_score'] = sascores
            scores_whole.extend(sascores.tolist())
            df.to_csv(f'{tmpdir}/{i}.tsv',sep='\t')
            paths.append(f'{tmpdir}/{i}.tsv')
        tsv_merge(paths,opath)
    return scores_whole


def descriptorcalculator_mod(smis, desc_list=descs_to_use, n_jobs=-1):
    assert n_jobs != 0
    calced = {}
    _j = n_jobs if n_jobs > 0 else os.cpu_count() + (n_jobs+1)
    def desc_parallel(smis,method,_j):
        met_list = lambda ms: np.array([method(Chem.MolFromSmiles(s)) for s in ms])
        _split = np.array_split(np.array(smis),_j)
        ret = Parallel(n_jobs=_j,backend='threading')(
            [delayed(met_list)(_s) for _s in _split]
        )
        return np.concatenate(ret)
    for desc in desc_list:
        calced[desc] = desc_parallel(smis,descs[desc],_j).ravel().tolist()
    return calced


def DescriptorCalculator(ipath, opath, smi_col='SMILES', desc_list=descs_to_use, n_jobs=-1):
    with TemporaryDirectory() as tmpdir:
        paths = []
        df_chunked  = pd.read_table(ipath,index_col=0,header=0,chunksize=chunk)
        for i, df in enumerate(df_chunked):
            calced = descriptorcalculator_mod(df[smi_col].to_numpy().ravel(),desc_list,n_jobs)
            for name, cal in calced.items():
                df[name] = cal
            if i==0:
                calced_whole = calced.copy()
            else:
                for name, cal in calced.items():
                    calced_whole[name] = calced_whole[name] + calced[name]
            df.to_csv(f'{tmpdir}/{i}.tsv',sep='\t')
            paths.append(f'{tmpdir}/{i}.tsv')
        tsv_merge(paths,opath)
    return calced_whole


def plot_kde(data, normalize_area=False, ax=None, **kwargs):
    """
    Function to plot a KDE for 1D data with an option to normalize the area.
    
    Parameters:
    - data: 1D data (list or NumPy array)
    - normalize_area: Whether to normalize the area of the plot (default: False)
    - ax: Matplotlib Axes object to plot on (if None, a new figure and axes will be created)
    - **kwargs: Other arguments passed to Seaborn's kdeplot
    
    Returns:
    - ax: The Axes object where the plot is drawn
    """
    # Create fig and ax if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if normalize_area:
        # Plot normalized KDE using Seaborn's kdeplot
        sns.kdeplot(data, ax=ax, **kwargs)
    else:
        # Plot KDE using gaussian_kde with the actual scale
        kde = gaussian_kde(list(data))
        x = np.linspace(min(data), max(data), 1000)
        ax.plot(x, kde(x) * len(data), **kwargs)

def dirnameextractor(name,confs):
    txt = f'{name}_level{confs["split_level"]}'
    if 'augmentation' in confs and confs['augmentation']:
        txt = f'{txt}_augmented'
    if "n_samples" not in confs:
        return txt, '_', '_'
    txt_sc = f'{txt}_{confs["n_samples"]}'
    if 'downsize_sc' in confs and confs['downsize_sc'] is not None:
        txt_sc = f'{txt_sc}_rc{confs["downsize_sc"]}'
    txt_ts = f'{txt}_{confs["n_samples"]}'
    if 'downsize_ts' in confs and confs['downsize_ts'] is not None:
        txt_ts = f'{txt_ts}_rc{confs["downsize_ts"]}'
    return txt, txt_sc, txt_ts

def most_frequent_element(lst):
    if not lst:
        return None  # リストが空の場合はNoneを返す

    counter = Counter(lst)
    most_common = counter.most_common(1)  # 最も頻度の高い要素を1つ取得

    return most_common[0][0], most_common[0][1]  # 要素とその出現回数をタプルで返す