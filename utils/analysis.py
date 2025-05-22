from collections import Counter
from copy import deepcopy
import json
import os
from tempfile import TemporaryDirectory
import warnings
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.stats import gaussian_kde
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles

from retrosynthesis import retrosep
from utils.utility import tsv_merge, dfconcatinatorwithlabel
from utils.chemutils import MurckoScaffoldSmilesListFromSmilesList, is_valid_molecule, MorganbitCalcAsVectors
from utils.SA_Score.sascorer import readFragmentScores, calculateScore
from models._kernel_and_mod import ProductTanimotoKernel
from models.modeling import ReactionGroupWrapperModeling
from utils.clustering import NearestNeighborSearchFromSmiles


chunk = 100000
fsize = 24


hist_seed_obj = lambda ax, x, y: ax.hist(x, histtype='step', bins=25, label=y)
hist_seed_hat = lambda ax, x, y: ax.hist(x, bins=25, label=y, alpha=0.5)

descs = {name: method for name, method in Descriptors.descList}
descs['SAscore'] = calculateScore
descs_to_use = ['MolWt', 'HeavyAtomCount', 'NumHDonors', 'NumHAcceptors', 'RingCount', 'TPSA', 'MolLogP','SAscore']

com_dict = {
        'svr_tanimoto_split': 'SVR-PK',
        'svr_tanimoto_average': 'SVR-SK',
        'svr_tanimoto': 'SVR-concatECFP'
    }
bas_dict = {'svr_tanimoto': 'SVR-baseline'}

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
    def read_and_process(file_path, reactant=True):
        data = pd.read_table(file_path, header=0, index_col=0).sort_index().replace('-', None).dropna()
        if reactant:
            data.replace('svr_tanimoto', 'svr_tanimoto_concat', inplace=True)
        return data

    res_rct_rxn_tr = read_and_process(f'{pred_dir}/prediction_score_rct_train.tsv')
    res_rct_rxn_ts = read_and_process(f'{pred_dir}/prediction_score_rct_test.tsv')
    res_prd_rxn_tr = read_and_process(f'{pred_dir}/prediction_score_prd_train.tsv', reactant=False)
    res_prd_rxn_ts = read_and_process(f'{pred_dir}/prediction_score_prd_test.tsv', reactant=False)

    res_rxn_tr = pd.concat([res_rct_rxn_tr, res_prd_rxn_tr])
    res_rxn_tr['model'] = [met[mod] for mod in res_rxn_tr['model']]
    res_rxn_ts = pd.concat([res_rct_rxn_ts, res_prd_rxn_ts])
    res_rxn_ts['model'] = [met[mod] for mod in res_rxn_ts['model']]

    return res_rxn_tr, res_rxn_ts

def is_augmented(confs, files, split_levels, rxns):
    """Process data for each split level and augmentation."""
    res_dict_tr_out = {}

    for split_level, split_label in split_levels.items():
        confs['split_level'] = split_level
        confs['augmentation'] = True
        pred_dir_base, _, _ = dirnameextractor('./outputs/prediction', confs)

        res_dict_tr = {}
        
        for file in files:
            file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
            pred_dir = os.path.join(pred_dir_base, file_uni_name)
            res_rxn_tr = pd.read_table(f'{pred_dir}/prediction_results_rct_train.tsv', header=0, index_col=0)

            # For each 'Rep_reaction', check if any entry has 'augmented' == True
            augmented_flags = res_rxn_tr.groupby('Rep_reaction')['augmented'].any()

            # Store processed data
            res_dict_tr[file_uni_name] = pd.DataFrame(augmented_flags)

        # Concatenate and process results
        res_df_tr = dfconcatinatorwithlabel(res_dict_tr, 'CHEMBL ID')
        res_df_tr['Identifier'] = [f'{id}_{rxn}' for id, rxn in zip(res_df_tr['CHEMBL ID'], res_df_tr.index)]
        res_df_tr['Reaction set ID'] = [rxns.loc[id, 'Reation set ID'] for id in res_df_tr['Identifier']]
        res_df_tr.sort_values(['Reaction set ID'], key=lambda s: [st.lower() if isinstance(st, str) else st for st in s], inplace=True)
        res_df_tr.set_index('Reaction set ID', inplace=True, drop=True)

        res_dict_tr_out[f'{split_label}_augmented'] = res_df_tr

    return res_dict_tr_out

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
            labels_origin = deepcopy(ax.get_xticklabels())
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
        ax.set_xticklabels(labels=labels_origin, fontsize=fsize, rotation=75)
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

def process_mlr_files_only(confs, files, prediction_levels, split_levels, rxns):
    """Process prediction files and organize results."""
    res_dict = {pslv: {} for pslv in prediction_levels}
    for file in files:
        file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
        for pslv, split_level in zip(prediction_levels, split_levels):
            confs['split_level'] = split_level
            pred_dir, _, _ = dirnameextractor('./outputs/prediction', confs)
            pred_dir = os.path.join(pred_dir, file_uni_name)
            res_prd_mlr = pd.read_csv(f'{pred_dir}/prd_molclr_scores.csv', header=0, index_col=None).sort_index()
            res_prd_mlr.rename(columns={'Unnamed: 0': 'dataset'}, inplace=True)
            rdict = {
                'MolCLR': res_prd_mlr,
            }
            rdf = dfconcatinatorwithlabel(rdict, 'split_level')
            res_dict[pslv][file_uni_name] = rdf.copy()
    dfs = {key: dfconcatinatorwithlabel(rd, 'uni_name').dropna() for key, rd in res_dict.items()}
    for key, df in dfs.items():
        df['identifier'] = [f'{idx}_{rxn}' for idx, rxn in zip(df['uni_name'], df['dataset'])]
        df['Reaction set ID'] = [rxns.loc[id, 'Reation set ID'] for id in df['identifier']]
        df.sort_values(by=['Reaction set ID'], inplace=True)
        dfs[key] = df
    return dfs

def process_mlr_files(confs, files, prediction_levels, split_levels):
    """Process prediction files and organize results."""
    res_dict = {pslv: {} for pslv in prediction_levels}
    for file in files:
        file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
        for pslv, split_level in zip(prediction_levels, split_levels):
            confs['split_level'] = split_level
            pred_dir, _, _ = dirnameextractor('./outputs/prediction', confs)
            pred_dir = os.path.join(pred_dir, file_uni_name)
            res_prd_mlr = pd.read_csv(f'{pred_dir}/prd_molclr_scores.csv', header=0, index_col=None).sort_index()
            res_prd_mlr.rename(columns={'Unnamed: 0': 'dataset', 'test_r2': 'r2', 'test_mae': 'mae', 'test_rmse': 'rmse'}, inplace=True)
            res_prd_svr = pd.read_table(f'{pred_dir}/prediction_score_prd_test.tsv', header=0, index_col=None).sort_index()
            res_prd_svr.rename(columns={'Rep_reaction': 'dataset', 'r2': 'r2'}, inplace=True)
            rdict = {
                'MolCLR': res_prd_mlr[['dataset', 'r2', 'mae', 'rmse']],
                'SVR': res_prd_svr[res_prd_svr['model'] == 'svr_tanimoto'][['dataset', 'r2', 'mae', 'rmse']]
            }
            rdf = dfconcatinatorwithlabel(rdict, 'split_level')
            res_dict[pslv][file_uni_name] = rdf.copy()
    return {key: dfconcatinatorwithlabel(rd, 'uni_name').dropna() for key, rd in res_dict.items()}

def plot_mlr_results(res_dfs, split_levels, output_path):
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
            labels_origin = deepcopy(ax.get_xticklabels())
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
        ax.set_xticklabels(labels=labels_origin, fontsize=fsize, rotation=75)
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
            labels_origin = deepcopy(ax.get_xticklabels())
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
        ax.set_xticklabels(labels=labels_origin, fontsize=fsize, rotation=75)
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
    """
    Perform statistical analysis on prediction results and return a formatted DataFrame.

    Args:
        confs (dict): Configuration dictionary.
        files (list): List of file paths.
        split_levels (list): List of split levels.
        eval_metric (str): Evaluation metric (e.g., 'r2').

    Returns:
        pd.DataFrame: Formatted DataFrame with statistical analysis results.
    """
    lab_dict = {1: 'Product-based', 2: 'Reactant-based'}

    results = []

    for file in files:
        file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
        row = {'Dataset': file_uni_name}

        for split_level in split_levels:
            confs['split_level'] = split_level
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

            # Perform statistical analysis
            for model, model_name in com_dict.items():
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

                p_value = stats.wilcoxon(
                    res_df_r['product'].to_list(),
                    res_df_r['reactant'].to_list(),
                    alternative="two-sided", mode="exact"
                ).pvalue

                col_name = f"{model_name} vs. {bas_dict['svr_tanimoto']} ({lab_dict[split_level]})"
                row[col_name] = p_value

        results.append(row)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).set_index('Dataset').T
    return results_df

def save_statistical_results(res_dict, output_path, eval_metric):
    """Save statistical analysis results to a TSV file."""
    res_df = pd.DataFrame(res_dict).T
    res_df.to_csv(f'{output_path}/wilcoxon_{eval_metric}_aug.tsv', sep='\t')
    print(f"Statistical results saved to {output_path}/wilcoxon_{eval_metric}_aug.tsv")
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
            res_rct_rxn_ts = res_rct_rxn_ts[~res_rct_rxn_ts['model'].str.startswith('_')]
            res_prd_rxn_ts = res_prd_rxn_ts[~res_prd_rxn_ts['model'].str.startswith('_')]

            # Combine and find the most frequent best model
            res_df_r = pd.concat([res_prd_rxn_ts, res_rct_rxn_ts]).astype({'r2': np.float64}).reset_index()
            res_idx = res_df_r.groupby('Rep_reaction')['r2'].idxmax()
            rlist.append(most_frequent_element(res_df_r.loc[res_idx]['model'].to_list())[0])

        # Count appearances
        res_dict[lab_dict[split_level]] = dict(Counter(rlist))

    # Save results
    res_df = pd.DataFrame(res_dict).T
    res_df.to_csv(output_path, sep='\t')
    print(f"Results saved to {output_path}")
    return res_df

def analyze_augmentation_effect(confs, files, split_levels, prediction_levels, com_dict):
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
    augmentation = [0, 1]

    for i, pslv in enumerate(prediction_levels):
        confs['split_level'] = split_levels[i]
        res_dict_mod = {c: 0 for c in com_dict.values()}
        res_dict_tar = {c: 0 for c in com_dict.values()}

        for file in files:
            file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
            confs_aug = confs.copy()
            confs['augmentation'] = 0
            confs_aug['augmentation'] = 1
            pred_dir, _, _ = dirnameextractor('./outputs/prediction', confs)
            pred_aug_dir, _, _ = dirnameextractor('./outputs/prediction', confs_aug)
            pred_dir = os.path.join(pred_dir, file_uni_name)
            pred_aug_dir = os.path.join(pred_aug_dir, file_uni_name)

            res_df_r = None
            for j, aug in enumerate(augmentation):
                confs['augmentation'] = aug
                pdir = pred_aug_dir if aug else pred_dir
                res_rct_rxn_ts = pd.read_table(
                    f'{pdir}/prediction_score_rct_test.tsv',
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

def plot_actual_vs_predicted(config_file_path, output_dir='./outputs/prediction', fsize=24):
    """
    For each file and reaction, generate and save scatter plots of measured vs. predicted values.
    The color indicates whether augmentation is used or not.
    """
    confs = load_config(config_file_path)

    files = confs["files"]
    obj_col = confs['objective_col']
    pred_rct_col = 'svr_tanimoto_split_rct_pred'
    pred_prd_col = 'svr_tanimoto_prd_pred'
    idx_col = confs['index_col']

    for file in files:
        file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
        axes_obj_dict = None
        fig = None

        for i, aug in enumerate([False, True]):
            confs['augmentation'] = aug
            pred_dir_base, _, _ = dirnameextractor(output_dir, confs)
            pred_dir = os.path.join(pred_dir_base, file_uni_name)

            prd_ts = pd.read_table(f'{pred_dir}/prediction_results_prd_test.tsv', header=0, index_col=0)
            rct_ts = pd.read_table(f'{pred_dir}/prediction_results_rct_test.tsv', header=0, index_col=0)

            reactions = sorted(set(rct_ts['Rep_reaction']))
            ver, hor = GridGenerator(reactions)

            if i == 0:
                # Create subplots for each reaction
                fig, axes = plt.subplots(ver, hor, figsize=(hor * 11, ver * 10))
                axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
                axes_obj_dict = {rxn: ax for rxn, ax in zip(reactions, axes)}

            for rxn, ax in axes_obj_dict.items():
                ax.set_xlabel('Measured $pK_i$', fontsize=fsize)
                ax.set_ylabel('Predicted $pK_i$', fontsize=fsize)
                if i == 0:
                    # Plot SVR-baseline and SVR-PK without augmentation for test data
                    ax.scatter(
                        prd_ts[prd_ts['Rep_reaction'] == rxn][obj_col],
                        prd_ts[prd_ts['Rep_reaction'] == rxn][pred_prd_col],
                        label='SVR-baseline', color='green'
                    )
                    ax.scatter(
                        rct_ts[rct_ts['Rep_reaction'] == rxn][obj_col],
                        rct_ts[rct_ts['Rep_reaction'] == rxn][pred_rct_col],
                        label='SVR-PK without augmentation', color='orange'
                    )
                else:
                    # Plot SVR-PK with augmentation for test data
                    ax.scatter(
                        rct_ts[rct_ts['Rep_reaction'] == rxn][obj_col],
                        rct_ts[rct_ts['Rep_reaction'] == rxn][pred_rct_col],
                        label='SVR-PK', color='blue'
                    )
                # Collect all relevant values for axis limits
                vals = []
                vals.extend(rct_ts[rct_ts['Rep_reaction'] == rxn][obj_col].values)
                vals.extend(rct_ts[rct_ts['Rep_reaction'] == rxn][pred_rct_col].values)
                vals.extend(prd_ts[prd_ts['Rep_reaction'] == rxn][pred_prd_col].values)
                lim = (min(vals) - 0.5, max(vals) + 0.5)

                ax.set_xlim(lim)
                ax.set_ylim(lim)
                
                # Draw diagonal line
                rlim_act = ax.get_xlim()
                rlim_obj = ax.get_ylim()
                rlim = [min(rlim_act[0], rlim_obj[0]), max(rlim_act[1], rlim_obj[1])]
                ax.plot([rlim[0], rlim[1]], [rlim[0], rlim[1]], color='red')
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=fsize)
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=fsize)
                ax.set_title(rxn, fontsize=fsize)
                ax.set_aspect('equal')

        # Add legends to each subplot
        for rxn, ax in axes_obj_dict.items():
            ax.legend(fontsize=18)

        # Set the main title and save the figure
        fig.suptitle(f'{file_uni_name}_actual_predict_plot', fontsize=fsize)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(f'{pred_dir}/prediction_results_plot.png')
        print(f"Scatter plot saved to {pred_dir}/prediction_results_plot.png")
        plt.clf()
        plt.close()

def grid_generator(unique):
    """Generate grid dimensions for subplots."""
    num_to_gen = len(unique)
    vertical = int(np.floor(np.sqrt(num_to_gen)))
    horizontal = int(np.ceil(num_to_gen / vertical))
    return vertical, horizontal

def process_and_plot_correlation(config_path, output_dir='./outputs/prediction', comps=1):
    """
    Process prediction results and plot correlations between kernel values and errors.

    Args:
        config_path (str): Path to the configuration file.
        output_dir (str): Base directory for prediction outputs.
        comps (int): Number of components for kernel calculation.
    """
    with open(config_path, 'r') as f:
        confs = json.load(f)

    files = confs["files"]
    idx_col = confs['index_col']
    obj_col = confs['objective_col']
    rct_obj_col = 'svr_tanimoto_split_rct_pred'
    rct_smi_col = 'Precursors'

    pred_dir_base, _, _ = dirnameextractor(output_dir, confs)

    for file in files:
        file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
        pred_dir = os.path.join(pred_dir_base, file_uni_name)

        # Load data
        rct_tr = pd.read_table(f'{pred_dir}/prediction_results_rct_train.tsv', header=0, index_col=0)
        rct_ts = pd.read_table(f'{pred_dir}/prediction_results_rct_test.tsv', header=0, index_col=0)

        # Generate grid for subplots
        uniques = set(rct_ts['Rep_reaction'])
        ver, hor = grid_generator(uniques)

        fig, axes = plt.subplots(ver, hor, figsize=(hor * 10, ver * 10))
        axes = list(axes.flatten()) if isinstance(axes, np.ndarray) else [axes]

        for i, (rxn, rxn_data_rct_ts) in enumerate(rct_ts.groupby('Rep_reaction')):
            rxn_data_rct_tr = rct_tr[rct_tr['Rep_reaction'] == rxn]
            rxn_data_rct_ts['abs_error'] = (rxn_data_rct_ts[obj_col] - rxn_data_rct_ts[rct_obj_col]).abs()

            # Calculate kernel values
            rxn_data_rct_tr_bits = csr_matrix(
                np.array(MorganbitCalcAsVectors(rxn_data_rct_tr[rct_smi_col], n_jobs=-1, split_components=True))
            )
            rxn_data_rct_ts_bits = csr_matrix(
                np.array(MorganbitCalcAsVectors(rxn_data_rct_ts[rct_smi_col], n_jobs=-1, split_components=True))
            )
            sim_mat = ProductTanimotoKernel(rxn_data_rct_ts_bits, rxn_data_rct_tr_bits)
            knl_val = np.mean(np.sort(sim_mat, axis=1)[:, ::-1][:, :comps], axis=1)
            rxn_data_rct_ts['knl_val'] = knl_val

            # Plot
            axes[i].scatter(rxn_data_rct_ts['knl_val'], rxn_data_rct_ts['abs_error'], color='blue')
            axes[i].set_xlim((0, 1))
            axes[i].set_title(f'{rxn}', fontsize=fsize)
            axes[i].set_xlabel('Maximum kernel value', fontsize=fsize)
            axes[i].set_ylabel('Absolute error value (SVR-PK)', fontsize=fsize)
            axes[i].set_xticklabels(labels=axes[i].get_xticklabels(), fontsize=fsize)
            axes[i].set_yticklabels(labels=axes[i].get_yticklabels(), fontsize=fsize)

        fig.suptitle(f'{file_uni_name} correlation of error and kernel value', fontsize=fsize)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(f'{pred_dir}/prediction_error.png')
        print(f"Correlation plot saved to {pred_dir}/prediction_error.png")

        plt.clf()
        plt.close()

def value_diffs_thres(sr, col_1, col_2, thres=1.0, over=False):
    """Check if the difference between two columns exceeds a threshold."""
    if over:
        return abs(sr[col_1] - sr[col_2]) >= thres
    return abs(sr[col_1] - sr[col_2]) < thres

def extract_large_errors(config_path, output_dir, diff_thres=0.5):
    """
    Extract large errors from prediction results and save the analysis.

    Args:
        config_path (str): Path to the configuration file.
        output_dir (str): Base directory for prediction outputs.
        diff_thres (float): Threshold for determining large errors.
    """
    confs = load_config(config_path)

    files = confs["files"]
    idx_col = confs['index_col']
    obj_col = confs['objective_col']
    prd_obj_col = 'svr_tanimoto_prd_pred'
    rct_obj_col = 'svr_tanimoto_product_rct_pred'
    split_level = confs['split_level']

    res_dict = {}
    for file in files:
        file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
        pred_dir, _, _ = dirnameextractor(output_dir, confs)

        # Load prediction data
        prd_tr = pd.read_table(f'{pred_dir}/prediction_results_prd_train.tsv', header=0, index_col=0)
        rct_tr = pd.read_table(f'{pred_dir}/prediction_results_rct_train.tsv', header=0, index_col=0)
        prd_ts = pd.read_table(f'{pred_dir}/prediction_results_prd_test.tsv', header=0, index_col=0)
        rct_ts = pd.read_table(f'{pred_dir}/prediction_results_rct_test.tsv', header=0, index_col=0)

        is_first = True
        for rxn, rxn_data_rct_ts in rct_ts.groupby('Rep_reaction'):
            is_first_inner = True
            rxn_data_prd_tr = prd_tr[prd_tr['Rep_reaction'] == rxn].copy()
            rxn_data_prd_ts = prd_ts[prd_ts['Rep_reaction'] == rxn].copy()
            rxn_data_rct_tr = rct_tr[rct_tr['Rep_reaction'] == rxn].copy()

            # Identify large errors
            rxn_data_rct_ts['is_over_error'] = rxn_data_rct_ts.apply(
                value_diffs_thres, axis=1, args=(obj_col, rct_obj_col, diff_thres, True)
            )
            rxn_data_prd_ts['is_not_over_error'] = rxn_data_prd_ts.apply(
                value_diffs_thres, axis=1, args=(obj_col, prd_obj_col, diff_thres, False)
            )
            rxn_data_rct_ts_oe = rxn_data_rct_ts[rxn_data_rct_ts['is_over_error']]

            for _, rxn_rct_mod in rxn_data_rct_ts_oe.iterrows():
                rxn_prd_mod = rxn_data_prd_ts[rxn_data_rct_ts[idx_col] == rxn_rct_mod[idx_col]].copy()
                assert rxn_prd_mod.shape[0] == 1
                if rxn_rct_mod['is_over_error'] and rxn_prd_mod['is_not_over_error'].iloc[0]:
                    rxn_rct_mod['Product_raw'] = rxn_prd_mod['Product_raw'].iloc[0]
                    rxn_rct_mod[prd_obj_col] = rxn_prd_mod[prd_obj_col].iloc[0]
                    rxn_rct_mod['is_not_over_error'] = rxn_prd_mod['is_not_over_error'].iloc[0]
                    data_for_analysis = pd.DataFrame(rxn_rct_mod).T if is_first_inner else pd.concat(
                        [data_for_analysis, pd.DataFrame(rxn_rct_mod).T]
                    )
                    is_first_inner = False

            if is_first_inner:
                continue

            # Perform nearest neighbor analysis
            nn_prd = NearestNeighborSearchFromSmiles()
            nn_prd.fit(rxn_data_prd_tr['Product_raw'].to_list())
            nn_dists, nn_idxs, nn_smis = nn_prd.transform(data_for_analysis['Product_raw'].to_list())
            data_for_analysis['nn_cpd_prd'] = nn_smis.ravel()
            data_for_analysis['nn_dist_prd'] = nn_dists.ravel()
            data_for_analysis['nn_actual_obj_prd'] = rxn_data_prd_tr.iloc[nn_idxs.ravel()][obj_col].to_list()

            nn_rct = NearestNeighborSearchFromSmiles(split_components=True)
            nn_rct.fit(rxn_data_rct_tr['Precursors'].to_list())
            nn_dists, nn_idxs, nn_smis = nn_rct.transform(data_for_analysis['Precursors'].to_list())
            data_for_analysis['nn_cpd_rct'] = nn_smis.ravel()
            data_for_analysis['nn_dist_rct'] = nn_dists.ravel()
            data_for_analysis['nn_actual_obj_rct'] = rxn_data_rct_tr.iloc[nn_idxs.ravel()][obj_col].to_list()

            data_for_analysis_whole = data_for_analysis.copy() if is_first else pd.concat(
                [data_for_analysis_whole, data_for_analysis]
            )
            is_first = False

        if is_first:
            continue

        # Save results
        data_for_analysis_whole.to_csv(f'{pred_dir}/error_analyzed.tsv', sep='\t')
        res_dict[file_uni_name] = data_for_analysis_whole.copy()

    # Save summary to Excel
    with pd.ExcelWriter(f'{output_dir}/prediction_level{split_level}/error_analyzed_summary.xlsx') as writer:
        for key, summary in res_dict.items():
            summary.to_excel(writer, sheet_name=key)

def plot_runtime_comparison(config_path, output_dir, rct_sizes, ids, fsize=20):
    """
    Plot runtime comparison between SVR-PK and Thompson sampling.

    Args:
        config_path (str): Path to the configuration file.
        output_dir (str): Base directory for output files.
        rct_sizes (list): List of reactant sizes to analyze.
        ids (list): List of dataset IDs for labeling.
        fsize (int): Font size for the plot.
    """
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap('tab20').colors)
    plt.rcParams["font.family"] = 'Nimbus Roman'

    confs = load_config(config_path)

    combs, tims_sc, tims_ts = [], [], []

    for rsize in rct_sizes:
        confs['downsize_sc'] = rsize
        _, dir_sc, _ = dirnameextractor(output_dir, confs)
        tim_sc = pd.read_table(f'{dir_sc}/svr-pk_screening_result_analysis.tsv', index_col='dataset', header=0)
        tim_ts = pd.read_table(f'{dir_sc}/thompson_screening_result_analysis.tsv', index_col='dataset', header=0)

        tims_sc.append(tim_sc['rct1_kernel_calctime'] + tim_sc['rct2_kernel_calctime'] +
                       tim_sc['evaluation_time'] + tim_sc['extraction_time'])
        tims_ts.append(tim_ts['sampling_time'])
        combs.append(tim_sc['possible_combinations'])

    idxs = list(range(len(tims_sc)))
    tims_sc_df = pd.DataFrame(tims_sc, index=idxs).T
    tims_ts_df = pd.DataFrame(tims_ts, index=idxs).T
    combs_df = pd.DataFrame(combs, index=idxs).T

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    for id, idx in zip(ids, tims_sc_df.index):
        ax.plot(combs_df.loc[idx], tims_sc_df.loc[idx], '-o', label=f'{id} SVR-PK', lw=3)
        ax.plot(combs_df.loc[idx], tims_ts_df.loc[idx], '-o', label=f'{id} Thompson', lw=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=fsize)
    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=fsize)
    ax.set_xlabel('Number of combinations', fontsize=fsize)
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=fsize)
    ax.set_ylabel('Execution time', fontsize=fsize)
    fig.tight_layout()

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap('tab10').colors)

def create_analysis_dataframe(config_path, output_dir, rxns_path):
    """
    Create a DataFrame based on the specified requirements and save it as a TSV file.

    Args:
        config_path (str): Path to the configuration file.
        output_dir (str): Base directory for output files.
        rxns_path (str): Path to the RXNs DataFrame.
    """
    # Load configuration and RXNs DataFrame
    confs = load_config(config_path)
    rxns = pd.read_table(rxns_path)
    rxns.set_index('Identifier', inplace=True)
    files = confs["files"]
    reactions = confs["reactions"]
    n_samples = confs["n_samples"]

    # Initialize results list
    results = []
    dir_preds_base, dir_sc_base, _ = dirnameextractor(output_dir, confs)

    for i, file in enumerate(files):
        file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
        rxn = reactions[i]
        print(f'Processing {file_uni_name} for reaction {rxn}')

        # Define paths
        dir_sc = f"{dir_sc_base}/{file_uni_name}/{rxn}"
        sc_preds_path = f'{dir_sc}/{file_uni_name}_{rxn}_rct_candidates_pairs_whole_sparse_split_highscored_retrieved_route.tsv'
        rct1s_path = f'{dir_sc}/{file_uni_name}_{rxn}_rct1_candidates_selected_whole.tsv'
        rct2s_path = f'{dir_sc}/{file_uni_name}_{rxn}_rct2_candidates_selected_whole.tsv'
        thompson_sc_path = f'{dir_sc}/ts_results_valid_route.tsv'

        # Extract Reaction set ID
        reaction_set_id = rxns.loc[f"{file_uni_name}_{rxn}", "Reation set ID"]

        # Initialize row data
        row_data = {"Reaction dataset ID": reaction_set_id, "Method": "SVR-PK"}
        sc_preds = pd.read_table(sc_preds_path, header=0, index_col=0)
        rct1s = pd.read_table(rct1s_path, header=0, index_col=0)
        rct1s.set_index('ID', drop=True, inplace=True)
        rct2s = pd.read_table(rct2s_path, header=0, index_col=0)
        rct2s.set_index('ID', drop=True, inplace=True)
        explored_combinations = len(rct1s) * len(rct2s)
        row_data["Explored combinations"] = f"{explored_combinations:.2e}"
        row_data["Screened products"] = n_samples
        row_data["Eligible products"] = sc_preds.shape[0]
        sc_scaf_whole = MurckoScaffoldSmilesListFromSmilesList(sc_preds['Product_norxncenter'].to_numpy().ravel()).ravel().tolist()
        sc_scaf_count = Counter(sc_scaf_whole)
        pd.DataFrame(sc_scaf_count.most_common(), columns=['scaf', 'appearance']).to_csv(
            f'{dir_sc}/sc_appearance.tsv', sep='\t'
        )
        print(f'Scaffolds appearance saved to {dir_sc}/sc_appearance.tsv')
        row_data["Unique scaffolds"] = len(set(sc_scaf_whole))
        row_data["Range of predicted pKi"] = f"{sc_preds['svr_tanimoto_predict'].min():.2e} - {sc_preds['svr_tanimoto_predict'].max():.2e}"
        sc_rct_pairs = [s.split(',') for s in sc_preds.index]
        sc_rct1 = [int(s[0]) for s in sc_rct_pairs]
        sc_rct2 = [int(s[1]) for s in sc_rct_pairs]
        sc_rct1s_extract = rct1s[rct1s.index.isin(sc_rct1)].copy()
        sc_rct2s_extract = rct2s[rct2s.index.isin(sc_rct2)].copy()
        sc_rct1s_extract['scaf'] = sc_rct1s_extract['SMILES'].apply(MurckoScaffoldSmilesFromSmiles)
        sc_rct2s_extract['scaf'] = sc_rct2s_extract['SMILES'].apply(MurckoScaffoldSmilesFromSmiles)
        row_data["Unique reactant1"] = len(set(sc_rct1))
        row_data["Unique reactant1 scaffolds"] = len(set(sc_rct1s_extract['scaf']))
        row_data["Unique reactant2"] = len(set(sc_rct2))
        row_data["Unique reactant2 scaffolds"] = len(set(sc_rct2s_extract['scaf']))

        # Process Thompson
        row_data_thompson = {"Reaction dataset ID": reaction_set_id, "Method": "Thompson"}
        thompson_sc = pd.read_table(thompson_sc_path, header=0, index_col=0)
        row_data_thompson["Explored combinations"] = "-"
        row_data_thompson["Screened products"] = n_samples
        row_data_thompson["Eligible products"] = thompson_sc.shape[0]
        ts_scaf_whole = MurckoScaffoldSmilesListFromSmilesList(thompson_sc['SMILES'].to_numpy().ravel()).ravel().tolist()
        ts_scaf_count = Counter(ts_scaf_whole)
        pd.DataFrame(ts_scaf_count.most_common(), columns=['scaf', 'appearance']).to_csv(
            f'{dir_sc}/ts_appearance.tsv', sep='\t'
        )
        print(f'Thompson sampling scaffolds appearance saved to {dir_sc}/ts_appearance.tsv')
        row_data_thompson["Unique scaffolds"] = len(set(ts_scaf_whole))
        row_data_thompson["Range of predicted pKi"] = f"{thompson_sc['score'].min():.2e} - {thompson_sc['score'].max():.2e}"
        ts_rct_pairs = [s.split('_') for s in thompson_sc.index]
        ts_rct1 = [int(s[0]) for s in ts_rct_pairs]
        ts_rct2 = [int(s[1]) for s in ts_rct_pairs]
        ts_rct1s_extract = rct1s[rct1s.index.isin(ts_rct1)].copy()
        ts_rct2s_extract = rct2s[rct2s.index.isin(ts_rct2)].copy()
        ts_rct1s_extract['scaf'] = ts_rct1s_extract['SMILES'].apply(MurckoScaffoldSmilesFromSmiles)
        ts_rct2s_extract['scaf'] = ts_rct2s_extract['SMILES'].apply(MurckoScaffoldSmilesFromSmiles)
        row_data_thompson["Unique reactant1"] = len(set(ts_rct1))
        row_data_thompson["Unique reactant1 scaffolds"] = len(set(ts_rct1s_extract['scaf']))
        row_data_thompson["Unique reactant2"] = len(set(ts_rct2))
        row_data_thompson["Unique reactant2 scaffolds"] = len(set(ts_rct2s_extract['scaf']))

        # Calculate intersections
        row_data["Intersections"] = len(set(sc_preds['Product_norxncenter']).intersection(thompson_sc['SMILES']))
        row_data_thompson["Intersections"] = row_data["Intersections"]

        # Append rows to results
        results.append(row_data)
        results.append(row_data_thompson)

    # Create DataFrame and save as TSV
    output_tsv = os.path.join(dir_sc_base, 'analysis_results.tsv')
    df = pd.DataFrame(results)
    df.to_csv(output_tsv, sep='\t', index=False)
    print(f"Results saved to {output_tsv}")

def plot_descriptor_boxplots(config_path, output_path, method='svr_tanimoto', fsize=20):
    """
    Generate and save boxplots for molecular descriptors.

    Args:
        config_path (str): Path to the configuration file.
        output_path (str): Path to save the output plot.
        method (str): Method name for prediction.
        fsize (int): Font size for the plot.
    """
    pred_col = f'{method}_predict'

    # Load configuration
    with open(config_path, 'r') as f:
        confs = json.load(f)

    files = confs["files"]
    reactions = confs["reactions"]

    # Prepare subplots
    num_descs = len(descs_to_use)
    rows = int(np.ceil(num_descs / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 5))
    axes = axes.flatten()

    res_dict = {}

    for i, file in enumerate(files):
        file_uni_name = os.path.split(file)[-1].rsplit('.', 1)[0]
        rxn = reactions[i]

        # Define paths
        _, dir_sc, _ = dirnameextractor('./outputs/reactant_combination', confs)
        dir_sc = f"{dir_sc}/{file_uni_name}/{rxn}"
        sc_preds_path = f'{dir_sc}/{file_uni_name}_{rxn}_rct_candidates_pairs_whole_sparse_split_highscored_retrieved_route.tsv'
        thompson_sc_path = f'{dir_sc}/ts_results_valid_route.tsv'

        # Load data
        try:
            sc_preds = pd.read_table(sc_preds_path, header=0, index_col=0)
        except:
            sc_preds = pd.DataFrame()

        try:
            thompson_sc = pd.read_table(thompson_sc_path, header=0, index_col=0)
            min_ts = np.min(thompson_sc['score'])
        except:
            thompson_sc = pd.DataFrame()
            min_ts = -np.inf

        # Filter predictions
        sc_preds_filtered = sc_preds[sc_preds[pred_col] >= min_ts] if not sc_preds.empty else pd.DataFrame()

        # Calculate descriptors
        rdict = {}
        try:
            rdict['SVR-PK'] = pd.concat([
                sc_preds_filtered[['Product_norxncenter']],
                pd.DataFrame(
                    DescriptorCalculator(
                        sc_preds_path,
                        f'{sc_preds_path.rsplit(".", 1)[0]}_descs.tsv',
                        'Product_norxncenter'
                    ),
                    index=sc_preds.index
                ).loc[sc_preds_filtered.index]
            ], axis=1).rename(columns={'Product_norxncenter': 'SMILES'})
        except:
            pass

        try:
            rdict['Thompson'] = pd.concat([
                thompson_sc[['SMILES']],
                pd.DataFrame(
                    DescriptorCalculator(
                        thompson_sc_path,
                        f'{thompson_sc_path.rsplit(".", 1)[0]}_descs.tsv',
                        'SMILES'
                    ),
                    index=thompson_sc.index
                )
            ], axis=1)
        except:
            pass

        if rdict:
            rdf = dfconcatinatorwithlabel(rdict, 'Method')
            res_dict[file_uni_name] = rdf

    # Combine results
    res_df = dfconcatinatorwithlabel(res_dict, 'uni_name')

    # Plot descriptors
    for i, desc in enumerate(descs_to_use):
        boxplotter(x='uni_name', y=desc, data=res_df, hue='Method', ax=axes[i])

    # Adjust plot settings
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=fsize)
        ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=fsize)

    fig.legend(handles=axes[0].get_legend_handles_labels()[0],
               labels=axes[0].get_legend_handles_labels()[1],
               fontsize=fsize, loc='lower center', ncols=len(axes[0].get_legend_handles_labels()[1]))
    fig.tight_layout()
    fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.95)
    fig.savefig(output_path)
    print(f"Boxplots saved to {output_path}")

    plt.clf()
    plt.close()

if __name__ == "__main__":
   print(1)

