from collections import Counter
import json
import os

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from tempfile import TemporaryDirectory
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns

from retrosynthesis import retrosep
from utils.utility import tsv_merge, dfconcatinatorwithlabel
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

def load_config(config_file_path):
    """Load configuration file."""
    with open(config_file_path, 'r') as f:
        return json.load(f)

def load_and_process_data(pred_dir, met):
    """Load and process prediction data."""
    def read_and_process(file_path):
        data = pd.read_table(file_path, header=0, index_col=0).sort_index().replace('-', None).dropna()
        data.replace('svr_tanimoto', 'svr_tanimoto_concat', inplace=True)
        return data

    res_rct_rxn_tr = read_and_process(f'{pred_dir}/prediction_score_rct_train.tsv')
    res_rct_rxn_ts = read_and_process(f'{pred_dir}/prediction_score_rct_test.tsv')
    res_prd_rxn_tr = read_and_process(f'{pred_dir}/prediction_score_prd_train.tsv')
    res_prd_rxn_ts = read_and_process(f'{pred_dir}/prediction_score_prd_test.tsv')

    res_rxn_tr = pd.concat([res_rct_rxn_tr, res_prd_rxn_tr])
    res_rxn_tr['model'] = [met[mod] for mod in res_rxn_tr['model']]
    res_rxn_ts = pd.concat([res_rct_rxn_ts, res_prd_rxn_ts])
    res_rxn_ts['model'] = [met[mod] for mod in res_rxn_ts['model']]

    return res_rxn_tr, res_rxn_ts

def process_split_level(confs, files, split_levels, augment, met, rxns):
    """Process data for each split level and augmentation."""
    res_dict_tr_out = {}
    res_dict_ts_out = {}

    for split_level, split_label in split_levels.items():
        for aug, aug_label in augment.items():
            confs['split_level'] = split_level
            confs['augmentation'] = aug
            pred_dir_base, _, _ = dirnameextractor('./outputs/prediction', confs)

            res_dict_tr = {}
            res_dict_ts = {}

            for file in files:
                file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
                pred_dir = os.path.join(pred_dir_base, file_uni_name)
                res_rxn_tr, res_rxn_ts = load_and_process_data(pred_dir, met)

                # Store processed data
                res_dict_tr[file_uni_name] = res_rxn_tr
                res_dict_ts[file_uni_name] = res_rxn_ts

            # Concatenate and process results
            res_df_tr = dfconcatinatorwithlabel(res_dict_tr, 'CHEMBL ID')
            res_df_tr['Identifier'] = [f'{id}_{rxn}' for id, rxn in zip(res_df_tr['CHEMBL ID'], res_df_tr.index)]
            res_df_tr['Reaction set ID'] = [rxns.loc[id, 'Reation set ID'] for id in res_df_tr['Identifier']]
            res_df_tr.drop(['CHEMBL ID', 'template', 'num_unique_scaf', 'data_statistics', 'Identifier'], axis=1, inplace=True)
            res_df_tr.sort_values(['Reaction set ID', 'model'], key=lambda s: [st.lower() if isinstance(st, str) else st for st in s], inplace=True)
            res_df_tr.set_index('Reaction set ID', inplace=True, drop=True)

            res_df_ts = dfconcatinatorwithlabel(res_dict_ts, 'CHEMBL ID')
            res_df_ts['Identifier'] = [f'{id}_{rxn}' for id, rxn in zip(res_df_ts['CHEMBL ID'], res_df_ts.index)]
            res_df_ts['Reaction set ID'] = [rxns.loc[id, 'Reation set ID'] for id in res_df_ts['Identifier']]
            res_df_ts.drop(['CHEMBL ID', 'template', 'num_unique_scaf', 'data_statistics', 'Identifier'], axis=1, inplace=True)
            res_df_ts.sort_values(['Reaction set ID', 'model'], key=lambda s: [st.lower() if isinstance(st, str) else st for st in s], inplace=True)
            res_df_ts.set_index('Reaction set ID', inplace=True, drop=True)

            res_dict_tr_out[f'{split_label}{aug_label}'] = res_df_tr
            res_dict_ts_out[f'{split_label}{aug_label}'] = res_df_ts

    return res_dict_tr_out, res_dict_ts_out

def save_to_excel(data_dict, output_path):
    """Save data dictionary to an Excel file."""
    with pd.ExcelWriter(output_path) as writer:
        for key, summary in data_dict.items():
            summary.to_excel(writer, sheet_name=key)

def process_files(confs, files, prediction_levels, split_levels):
    """Process prediction files and organize results."""
    res_dict = {pslv: {} for pslv in prediction_levels}
    for file in files:
        file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
        for pslv, split_level in zip(prediction_levels, split_levels):
            confs['split_level'] = split_level
            pred_dir, _, _ = dirnameextractor('./outputs/prediction', confs)
            pred_dir = os.path.join(pred_dir, file_uni_name)
            res_rct_rxn_ts = pd.read_table(f'{pred_dir}/prediction_score_rct_test.tsv', header=0, index_col=None).sort_index()
            res_prd_rxn_ts = pd.read_table(f'{pred_dir}/prediction_score_prd_test.tsv', header=0, index_col=None).sort_index()
            rdict = {
                'SVR-PK': res_rct_rxn_ts[res_rct_rxn_ts['model'] == 'svr_tanimoto_split'],
                'SVR-SK': res_rct_rxn_ts[res_rct_rxn_ts['model'] == 'svr_tanimoto_average'],
                'SVR-concatECFP': res_rct_rxn_ts[res_rct_rxn_ts['model'] == 'svr_tanimoto'],
                'SVR-baseline': res_prd_rxn_ts[res_prd_rxn_ts['model'] == 'svr_tanimoto']
            }
            rdf = dfconcatinatorwithlabel(rdict, 'split_level')
            res_dict[pslv][file_uni_name] = rdf.copy()
    return {key: dfconcatinatorwithlabel(rd, 'uni_name').dropna() for key, rd in res_dict.items()}

def plot_results(res_dfs, split_levels, output_path):
    """Generate and save boxplots for the results."""
    fsize = 30
    lab_dict = {1: 'Product-based', 2: 'Reactant-based'}
    met = {'r2': '$R^2$', 'mae': 'MAE'}
    
    rfig, rax = plt.subplots(2, len(split_levels), figsize=(len(split_levels) * 12, 20))
    for i, (key, df) in enumerate(res_dfs.items()):
        boxplotter(x='uni_name', y='r2', data=df, hue='split_level', ax=rax[0, i])
        boxplotter(x='uni_name', y='mae', data=df, hue='split_level', ax=rax[1, i])
    
    for i, a in enumerate(rax):
        for j, ax in enumerate(a):
            if i == 0:
                ax.set_title(lab_dict[split_levels[j]], fontsize=fsize * 2)
            ax_lim = ax.get_ylim()
            if ax_lim[0] > ax_lim[1]:
                ax.invert_yaxis()
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1, symbol=''))
            ax.set_xticklabels(labels=["" for _ in ax.get_xticklabels()])
            ax.set_xlabel("")
            if j == 0:
                ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=fsize)
                ax.set_ylabel(met[ax.get_ylabel()], fontsize=fsize + 20)
            else:
                ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=fsize)
                ax.set_ylabel("")
    
    for ax in rax[-1]:
        ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=fsize, rotation=75)
        ax.set_xlabel("CHEMBL ID", fontsize=fsize + 10)
    
    hdl, lbls = ax.get_legend_handles_labels()
    for ax in rax.ravel():
        ax.get_legend().remove()
    
    rfig.legend(handles=hdl, labels=lbls, fontsize=fsize, loc='lower center', ncols=len(lbls))
    rfig.tight_layout()
    rfig.subplots_adjust(left=0.08, bottom=0.15, right=0.99, top=0.95)
    rfig.savefig(output_path)
    plt.clf()
    plt.close()

def process_augmentation_differences(confs, files, split_levels, prediction_levels):
    """Process differences in augmentation results."""
    res_dict = {pslv: {} for pslv in prediction_levels}
    for file in files:
        file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
        for pslv, split_level in zip(prediction_levels, split_levels):
            rdiff = {}
            for j, aug in enumerate([True, False]):
                confs['augmentation'] = aug
                confs['split_level'] = split_level
                pred_dir, _, _ = dirnameextractor('./outputs/prediction', confs)
                pred_dir = os.path.join(pred_dir, file_uni_name)
                res_rct_rxn_ts = pd.read_table(f'{pred_dir}/prediction_score_rct_test.tsv', header=0, index_col=None).sort_index()
                res_prd_rxn_ts = pd.read_table(f'{pred_dir}/prediction_score_prd_test.tsv', header=0, index_col=None).sort_index()
                rdict = {
                    'SVR-PK': res_rct_rxn_ts[res_rct_rxn_ts['model'] == 'svr_tanimoto_split'],
                    'SVR-SK': res_rct_rxn_ts[res_rct_rxn_ts['model'] == 'svr_tanimoto_average'],
                    'SVR-concatECFP': res_rct_rxn_ts[res_rct_rxn_ts['model'] == 'svr_tanimoto']
                }
                rdf = dfconcatinatorwithlabel(rdict, 'split_level')
                rdiff[j] = rdf
            rdiff_df = rdiff[0][['r2', 'rmse', 'mae']].astype(float) - rdiff[1][['r2', 'rmse', 'mae']].astype(float)
            rdiff_df['rmse'] = -rdiff_df['rmse']
            rdiff_df['mae'] = -rdiff_df['mae']
            rdiff_df['split_level'] = rdiff[0]['split_level']
            res_dict[pslv][file_uni_name] = rdiff_df.copy()
    return {key: dfconcatinatorwithlabel(rd, 'uni_name').dropna() for key, rd in res_dict.items()}

def plot_augmentation_differences(res_dfs, split_levels, output_path):
    """Generate and save boxplots for augmentation differences."""
    fsize = 30
    lab_dict = {1: 'Product-based', 2: 'Reactant-based'}
    met = {'r2': '$R^2$', 'mae': 'MAE'}
    
    rfig, rax = plt.subplots(2, len(split_levels), figsize=(len(split_levels) * 12, 20))
    for i, (key, df) in enumerate(res_dfs.items()):
        boxplotter(x='uni_name', y='r2', data=df, hue='split_level', ax=rax[0, i])
        boxplotter(x='uni_name', y='mae', data=df, hue='split_level', ax=rax[1, i])
    
    for i, a in enumerate(rax):
        for j, ax in enumerate(a):
            if i == 0:
                ax.set_title(lab_dict[split_levels[j]], fontsize=fsize * 2)
            ax_lim = ax.get_ylim()
            if ax_lim[0] > ax_lim[1]:
                ax.invert_yaxis()
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1, symbol=''))
            ax.set_xticklabels(labels=["" for _ in ax.get_xticklabels()])
            ax.set_xlabel("")
            if j == 0:
                ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=fsize)
                ax.set_ylabel(met[ax.get_ylabel()], fontsize=fsize + 20)
            else:
                ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=fsize)
                ax.set_ylabel("")
    
    for ax in rax[-1]:
        ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=fsize, rotation=75)
        ax.set_xlabel("CHEMBL ID", fontsize=fsize + 10)
    
    hdl, lbls = ax.get_legend_handles_labels()
    for ax in rax.ravel():
        ax.get_legend().remove()
    
    rfig.legend(handles=hdl, labels=lbls, fontsize=fsize, loc='lower center', ncols=len(lbls))
    rfig.tight_layout()
    rfig.subplots_adjust(left=0.10, bottom=0.15, right=0.99, top=0.95)
    rfig.savefig(output_path)
    plt.clf()
    plt.close()

def process_statistical_analysis(confs, files, split_levels, eval_metric):
    """Perform statistical analysis on prediction results."""
    prediction_levels = [f'prediction_level{spl}' for spl in split_levels]
    res_dict = {pslv: {} for pslv in prediction_levels}
    com_dict = {
        'svr_tanimoto_split': 'SVR-PK',
        'svr_tanimoto_average': 'SVR-SK',
        'svr_tanimoto': 'SVR-concatECFP'
    }
    bas_dict = {'svr_tanimoto': 'SVR-baseline'}

    for file in files:
        file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
        for pslv, split_level in zip(prediction_levels, split_levels):
            confs['split_level'] = split_level
            pred_dir, _, _ = dirnameextractor('./outputs/prediction', confs)
            pred_dir = os.path.join(pred_dir, file_uni_name)

            res_rct_rxn_ts = pd.read_table(
                f'{pred_dir}/prediction_score_rct_test.tsv',
                header=0, index_col=None
            ).sort_index().replace('-', None).dropna()

            res_prd_rxn_ts = pd.read_table(
                f'{pred_dir}/prediction_score_prd_test.tsv',
                header=0, index_col=None
            ).sort_index().replace('-', None).dropna()

            rlist = []
            for model in com_dict.keys():
                res_df_r = pd.concat([
                    res_prd_rxn_ts[res_prd_rxn_ts['model'] == 'svr_tanimoto']
                    .set_index('Rep_reaction')[[eval_metric]]
                    .rename(columns={eval_metric: 'product'})
                    .astype(np.float64),
                    res_rct_rxn_ts[res_rct_rxn_ts['model'] == model]
                    .set_index('Rep_reaction')[[eval_metric]]
                    .rename(columns={eval_metric: 'reactant'})
                    .astype(np.float64)
                ], axis=1)
                rlist.append(stats.wilcoxon(
                    res_df_r['product'].to_list(),
                    res_df_r['reactant'].to_list(),
                    alternative="two-sided", mode="exact"
                ).pvalue)
            res_dict[pslv][file_uni_name] = rlist
    return res_dict

def save_statistical_results(res_dict, output_path, eval_metric):
    """Save statistical analysis results to a TSV file."""
    res_df = pd.DataFrame(res_dict).T
    res_df.to_csv(f'{output_path}/wilcoxon_{eval_metric}_aug.tsv', sep='\t')
    return res_df

def analyze_best_model_appearance(confs, files, split_levels, com_dict, bas_dict, output_path):
    """
    Analyze the appearance of the best models for each reaction and save the results.

    Args:
        confs (dict): Configuration dictionary.
        files (list): List of file paths.
        split_levels (list): List of split levels.
        com_dict (dict): Dictionary mapping model keys to names.
        bas_dict (dict): Dictionary mapping baseline model keys to names.
        output_path (str): Path to save the results.
    """
    lab_dict = {1: 'Product-based', 2: 'Reactant-based'}
    res_dict = {}

    for split_level in split_levels:
        confs['split_level'] = split_level
        rlist = []

        for file in files:
            file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
            pred_dir, _, _ = dirnameextractor('./outputs/prediction', confs)
            pred_dir = os.path.join(pred_dir, file_uni_name)

            # Load prediction results
            res_rct_rxn_ts = pd.read_table(
                f'{pred_dir}/prediction_score_rct_test.tsv',
                header=0, index_col=None
            ).sort_index().replace('-', None).dropna()

            res_prd_rxn_ts = pd.read_table(
                f'{pred_dir}/prediction_score_prd_test.tsv',
                header=0, index_col=None
            ).sort_index().replace('-', None).dropna()

            # Map model names
            res_rct_rxn_ts['model'] = res_rct_rxn_ts['model'].map(com_dict)
            res_prd_rxn_ts['model'] = res_prd_rxn_ts['model'].map(bas_dict)

            # Combine and find the most frequent best model
            res_df_r = pd.concat([res_prd_rxn_ts, res_rct_rxn_ts]).astype({'r2': np.float64}).reset_index()
            res_idx = res_df_r.groupby('Rep_reaction')['r2'].idxmax()
            rlist.extend(most_frequent_element(res_df_r.loc[res_idx]['model'].to_list()))

        # Count appearances
        res_dict[lab_dict[split_level]] = dict(Counter(rlist))

    # Save results
    res_df = pd.DataFrame(res_dict).T
    res_df.to_csv(output_path, sep='\t')
    print(f"Results saved to {output_path}")

def analyze_augmentation_effect(confs, files, split_levels, prediction_levels, augmentation, com_dict):
    """
    Analyze the effect of augmentation on prediction results.

    Args:
        confs (dict): Configuration dictionary.
        files (list): List of file paths.
        split_levels (list): List of split levels.
        prediction_levels (list): List of prediction levels.
        augmentation (list): List of augmentation flags.
        com_dict (dict): Dictionary mapping model keys to names.

    Returns:
        dict: Results of the analysis.
    """
    results = {}

    for i, pslv in enumerate(prediction_levels):
        confs['split_level'] = split_levels[i]
        res_dict_mod = {c: 0 for c in com_dict.values()}
        res_dict_tar = {c: 0 for c in com_dict.values()}

        for file in files:
            file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
            pred_dir, _, _ = dirnameextractor('./outputs/prediction', confs)
            pred_dir = os.path.join(pred_dir, file_uni_name)

            res_df_r = None
            for j, aug in enumerate(augmentation):
                confs['augmentation'] = aug
                res_rct_rxn_ts = pd.read_table(
                    f'{pred_dir}/prediction_score_rct_test.tsv',
                    header=0, index_col=None
                ).sort_index().replace('-', None).dropna()

                res_rct_rxn_ts['model'] = res_rct_rxn_ts['model'].map(com_dict)
                res_rct_rxn_ts = res_rct_rxn_ts.astype({'r2': np.float64}).rename(columns={'r2': aug})
                res_df_r = res_rct_rxn_ts[['Rep_reaction', 'model', aug]] if j == 0 else pd.concat([res_df_r, res_rct_rxn_ts[[aug]]], axis=1)

            res_df_r['diff'] = res_df_r[1] - res_df_r[0]
            for c in res_dict_mod.keys():
                res_dict_mod[c] += res_df_r[(res_df_r['model'] == c) & (res_df_r['diff'] > 0)].shape[0]
                if res_df_r[(res_df_r['model'] == c) & (res_df_r['diff'] > 0)].shape[0] > res_df_r[(res_df_r['model'] == c) & (res_df_r['diff'] != 0)].shape[0] / 2:
                    res_dict_tar[c] += 1

        results[pslv] = {'mod': res_dict_mod, 'tar': res_dict_tar}

    return results

def analyze_and_plot_augmentation_effect(config_file_path, output_path):
    """
    Analyze the effect of augmentation and plot the results.

    Args:
        config_file_path (str): Path to the configuration file.
        output_path (str): Path to save the output plot.
    """
    # Load configuration
    confs = load_config(config_file_path)
    files = confs["files"]
    split_levels = [1, 2]
    prediction_levels = [f'prediction_level{spl}' for spl in split_levels]

    # Process augmentation differences
    res_dfs = process_augmentation_differences(confs, files, split_levels, prediction_levels)

    # Plot results
    plot_augmentation_differences(res_dfs, split_levels, output_path)
    print(f"Augmentation effect analysis saved to {output_path}")

if __name__ == "__main__":
   print(1)

