''' Thompson sampling
'''

import os,json,sys
pwd = os.path.dirname(os.path.abspath(__file__))
# os.chdir(pwd)
sys.path.append(pwd)
sys.path.append(f'{pwd}/_benchmarking/TS_main_20240607')

import pandas as pd
import numpy as np
from rdkit import Chem
import time
from time import time
from scipy.sparse import csr_matrix,vstack
import pickle
import argparse
import tempfile
import swifter
from joblib import Parallel, delayed, cpu_count

from models._kernel_and_mod import funcTanimotoSklearn
from models.modeling import *
from screening.screening_mod import ReactantScreening
from utils.utility import tsv_merge, logger, MakeDirIfNotExisting, AttrJudge, timer
from utils.chemutils import ReactionCenter, reactor, is_valid_molecule, MorganbitCalcAsVectorFromSmiles, SmilesExtractor
from utils.analysis import ValidityAndSuggestedRouteExaminator
from _benchmarking.TS_main_20240607.ts_logger import get_logger


CHUNK = 100000
rng = np.random.default_rng(seed = 0)

parser = argparse.ArgumentParser(description='Collect reactants and combine them...')
parser.add_argument('-c', '--config', default=f'{pwd}/config/example_config.json',
    help='Configration')

args = parser.parse_args()

## Read configurations from json file
confs = json.load(open(args.config,'r'))
# Settings used in this program
files            = confs['files']
if 'cand_path' not in confs:
    confs['cand_path'] = f'{pwd}/emolecules/emolecules_compounds_curated.tsv'
confs['cand_path'] = os.path.abspath(confs['cand_path'])
cand_path        = confs['cand_path']
out_dir          = AttrJudge(confs, 'out_dir', f'{pwd}/outputs') 
# ext_ratio        = AttrJudge(confs, 'ext_ratio', 1e-5) 
reactions        = confs['reactions']
split_level      = confs['split_level']
n_samples        = confs['n_samples']
augmentation     = AttrJudge(confs, 'augmentation', False)
downsize         = AttrJudge(confs, 'downsize_ts', None)
precalc          = AttrJudge(confs, 'precalc', False)
postpro          = AttrJudge(confs, 'postprocess', False)

def importstr(module_str, from_=None):
	"""
	module_str: module to be loaded as string 
	>>> importstr('os) -> <module 'os'>
	"""
	if (from_ is None) and ':' in module_str:
		module_str, from_ = module_str.rsplit(':')
	module = __import__(module_str)
	for sub_str in module_str.split('.')[1:]:
		module = getattr(module, sub_str)
	
	if from_:
		try:
			return getattr(module, from_)
		except:
			raise ImportError(f'{module_str}.{from_}')
	return module


def run(app, *argv):
    argv=list(argv)
    app_cls=importstr(app)
    sys.argv = [sys.argv[0]]
    for arg in argv:
        sys.argv.append(arg)
    app_cls.main()


def SmilesExtractor(tpath,smiles_col,idx_col,opath):
    df = pd.read_table(tpath,header=0,index_col=0,chunksize=CHUNK)
    with open(opath,'w') as of:
        df_chunked = next(df)
        if not df_chunked.empty:
            idx_list = df_chunked[idx_col].to_list()
            smi_list = df_chunked[smiles_col].to_list()
            joined   = [f'{smi}\t{idx}' for idx, smi in zip(idx_list,smi_list)]
            joined.append('')
            smi_str  = '\n'.join(joined)
            of.write(smi_str)
    with open(opath,'a') as of:
        for df_chunked in df:
            if not df_chunked.empty:
                idx_list = df_chunked[idx_col].to_list()
                smi_list = df_chunked[smiles_col].to_list()
                joined   = [f'{smi}\t{idx}' for idx, smi in zip(idx_list,smi_list)]
                joined.append('')
                smi_str  = '\n'.join(joined)
                of.write(smi_str)

if augmentation:
    odir_pref = f'{out_dir}/reactant_combination_level{split_level}_augmented'
else:
    odir_pref = f'{out_dir}/reactant_combination_level{split_level}'

if downsize is not None:
    out_dir_sc_base  = f'{odir_pref}_{n_samples}_rc{downsize}'
else:
    out_dir_sc_base  = f'{odir_pref}_{n_samples}'

if isinstance(reactions, str):
    reactions = [reactions for _ in range(len(files))]
assert(len(files)==len(reactions))

def main():
    rdicts = []
    for odr, file in enumerate(files):
        file_uni_name = os.path.split(file)[-1].rsplit('.',1)[0]
        reaction = reactions[odr]
        MakeDirIfNotExisting(f"{out_dir_sc_base}/{file_uni_name}/{reaction}")
        lgr_name = f"{out_dir_sc_base}/{file_uni_name}/{reaction}/ts_logs.txt"
        lgr = logger(filename=lgr_name)
        lgr.write(f'<<< Start reactant combination of {file_uni_name} >>>')
        lgr.write('======Config sets======')
        lgr.write(f'file: {file}')
        lgr.write(f'reaction: {reaction}')
        [lgr.write(f'{key}: {val}') for key, val in confs.items() if key not in ['files','reactions']]
        lgr.write('=======================')

        rs = ReactantScreening(file_uni_name, reaction, lgr, confs)
        
        if not precalc:
            cands = pd.read_table(cand_path,header=0,index_col=0, chunksize=CHUNK)
            arg_rct1 = (1,True,)
            arg_rct2 = (2,True,)
            rs.SubmatchAndMapping(cands, arg_rct1, arg_rct2)
    
        opath_rct1 = f'{rs.dir_cand}/{file_uni_name}_{reaction}_rct1_candidates_smiles.smi'
        opath_rct2 = f'{rs.dir_cand}/{file_uni_name}_{reaction}_rct2_candidates_smiles.smi'

        if downsize is not None:
            lgr.write(f'Downsize option is selected. Size: {downsize}')
            dpath_rct1 = rs.randomselector(rs.mp_rct1,'downsize_ts')
            dpath_rct2 = rs.randomselector(rs.mp_rct2,'downsize_ts')

        SmilesExtractor(dpath_rct1,'washed_isomeric_kekule_smiles','EMOL_VERSION_ID',opath_rct1)
        SmilesExtractor(dpath_rct2,'washed_isomeric_kekule_smiles','EMOL_VERSION_ID',opath_rct2)

        ts_dict = {
            "reagent_file_list": [
                opath_rct1,
                opath_rct2
            ],
            "reaction_smarts": rs.rc.template,
            "num_warmup_trials": 3,
            "num_ts_iterations": n_samples,
            "evaluator_class_name": "ObjectiveEvaluatorByTanimotoKernel",
            "evaluator_arg": {
                "mod_path": rs.model_path,
                "reaction_metaname": reaction
            },
            "ts_mode": "maximize",
            "log_filename": lgr_name,
            "results_filename": f"{rs.dir_cand}/ts_results.csv"
                }
        
        with open(f'{rs.dir_cand}/ts_envs.json','w') as f:
            json.dump(ts_dict,f)

        with timer() as t:
            run('_benchmarking.TS_main_20240607.ts_main',
                f'{rs.dir_cand}/ts_envs.json')
            rs.res_dict['sampling_time'] = t.get_runtime()
        
        lgr = get_logger(__name__,filename=ts_dict['log_filename'])
        lgr.info('Start validation of extracted compounds !')
        start = time()
        with tempfile.TemporaryDirectory() as tmpdir:
            sh = 0
            dfs = pd.read_csv(f'{rs.dir_cand}/ts_results.csv',index_col=None,header=0,chunksize=CHUNK)
            valid_list = []
            mod_is_valid_mol = lambda x: [is_valid_molecule(x_) for x_ in x]
            for i,df in enumerate(dfs):
                smiles_lists = np.array_split(df['SMILES'].to_numpy(),os.cpu_count())
                valid_lists  = Parallel(n_jobs=-1,backend='threading')(
                    [delayed(mod_is_valid_mol)(smiles_list)
                    for smiles_list in smiles_lists]
                )
                df['valid']  = [x for row in valid_lists for x in row]
                df[df['valid']].to_csv(f'{tmpdir}/{i}.tsv',sep='\t')
                sh += df[df['valid']].shape[0]
                valid_list.append(f'{tmpdir}/{i}.tsv')
            tsv_merge(valid_list,f'{rs.dir_cand}/ts_results_valid.tsv')
            rs.res_dict['combinations_cpd_filter'] = sh
        lgr.info(f'End validation. Took {time()-start} seconds.')

        if postpro:
            lgr.info('Checking the suggested route...')
            lgr.info('This process may takes long.')
            p, shape = ValidityAndSuggestedRouteExaminator(f'{rs.dir_cand}/ts_results_valid.tsv',f'{rs.dir_cand}/ts_results_valid_route.tsv',
                                                input_smi_col='SMILES',prec_smi_col=None,ext_data_pathes=[rs.mp_rct1,rs.mp_rct2],input_index_col='Name',
                                                ext_index_col='EMOL_VERSION_ID',ext_smi_col='mapped_smiles',ext_index_sep='_')
            rs.res_dict['combinations_retro_filter'] = shape
            lgr.info('Check completed.')
            lgr.info(f'{shape} compounds are remained.')
        rdicts.append(rs.res_dict.copy())
    rdf = pd.DataFrame(rdicts)
    rdf.to_csv(f'{out_dir_sc_base}/thompson_screening_result_analysis.tsv',sep='\t')

if __name__ == '__main__':
    main()
