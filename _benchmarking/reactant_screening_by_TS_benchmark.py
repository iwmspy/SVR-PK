''' Thompson sampling
'''

import os,json,sys
pwd_deep = os.path.dirname(os.path.abspath(__file__))
pwd = os.path.abspath(os.path.join(pwd_deep,'..'))
# os.chdir(pwd)
sys.path.append(pwd)
sys.path.append(f'{pwd_deep}/TS_main_20240607')

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
from utils.utility import mean_of_top_n_elements, tsv_merge, mkdir, timer
from utils.chemutils import ReactionCenter, reactor, is_valid_molecule, MorganbitCalcAsVectorFromSmiles
from _benchmarking.TS_main_20240607.ts_logger import get_logger

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


chunk = 100000
rng = np.random.default_rng(seed = 0)

parser = argparse.ArgumentParser(description='Collect reactants and combine them...')
parser.add_argument('-c', '--config', default=f'{pwd}/config/example_config.json',
    help='Configration')

args = parser.parse_args()

## Read configurations from json file
confs = json.load(open(args.config,'r'))
# Settings used in this program
files = confs['files']
index_col = confs['index_col']
model_smiles_col = confs['model_smiles_col']
objective_col = confs['objective_col']
cand_path = os.path.abspath(confs['cand_path']) if 'cand_path' in confs else \
    f'{pwd}/emolecule/emolecule_compounds_curated.tsv'
only_highscored = confs['only_highscored'] if 'only_highscored' in confs else True
reactions = confs['reactions']
split_level  = confs['split_level']
downsize = confs['downsize'] if 'downsize' in confs else None
precalc = confs['precalc'] if 'precalc' in confs else True

# assert(split_option in ('random','similarity'))

def LoadMetadatas(file_uni_name,reaction):
    path_fr_ecfp  = f'{pwd}/outputs/prediction_level{split_level}/{file_uni_name}/prediction_results_rct_test.tsv'
    path_prd_ecfp = f'{pwd}/outputs/prediction_level{split_level}/{file_uni_name}/prediction_results_prd_train.tsv'
    ## metadata
    tid_ecfp_fr_tr  = pd.read_table(path_fr_ecfp, header=0, index_col=0)
    tid_ecfp_prd_tr = pd.read_table(path_prd_ecfp, header=0, index_col=0)
    return tid_ecfp_fr_tr[tid_ecfp_fr_tr['Rep_reaction']==reaction], tid_ecfp_prd_tr[tid_ecfp_prd_tr['Rep_reaction']==reaction]

def LoadKernels(model_path):
    # load model 
    rxnmlmod = pickle.load(open(model_path, 'rb'))
    return rxnmlmod


# dir_analysis = './outputs/analysis/ecfp'
for odr, file in enumerate(files):
    file_uni_name = os.path.split(file)[-1].rsplit('.',1)[0]
    reaction = reactions[odr]
    model_path    = f'{pwd}/outputs/prediction_level{split_level}/{file_uni_name}/mod.pickle'
    dir_cand = f'{pwd_deep}/outputs/reactant_combination_level{split_level}/{file_uni_name}/{reaction}'
    mp_rct1  = f'{dir_cand}/{file_uni_name}_{reaction}_rct1_candidates_whole.tsv'
    mps_rct1 = f'{dir_cand}/{file_uni_name}_{reaction}_rct1_candidates_selected_whole.tsv'
    mpk_rct1 = f'{dir_cand}/{file_uni_name}_{reaction}_rct1_candidates_selected_kernel_whole.npy'
    mp_rct2  = f'{dir_cand}/{file_uni_name}_{reaction}_rct2_candidates_whole.tsv'
    mps_rct2 = f'{dir_cand}/{file_uni_name}_{reaction}_rct2_candidates_selected_whole.tsv'
    mpk_rct2 = f'{dir_cand}/{file_uni_name}_{reaction}_rct2_candidates_selected_kernel_whole.npy'

    md = mkdir()
    rxnmlmod = LoadKernels(model_path=model_path)

    test_rct, train_prd = LoadMetadatas(file_uni_name, reaction)

    template = test_rct[test_rct['Rep_reaction']==reaction]['template'].iloc[0]
    rc = reactor(template,cumurate=False)

    sub = template.split('>>')[1]
    sub_rct1,sub_rct2 = sub.split('.')
    
    rct1_pool = set(test_rct['Precursors1'])
    rct2_pool = set(test_rct['Precursors2'])
    isin_rct1_pool = lambda x: x in rct1_pool
    isin_rct2_pool = lambda x: x in rct2_pool

    center = ReactionCenter(sub,cumurate=False)

    submol_rct1 = Chem.MolFromSmarts(sub_rct1)
    submol_rct2 = Chem.MolFromSmarts(sub_rct2)

    rct1_hatoms_range = (np.min(test_rct['Precursors1_num_of_hatoms']),
                        np.max(test_rct['Precursors1_num_of_hatoms']))
    rct2_hatoms_range = (np.min(test_rct['Precursors2_num_of_hatoms']),
                        np.max(test_rct['Precursors2_num_of_hatoms']))

    def LoadKernelVals():
        # load kernels
        kernel_fr1 = np.load(mpk_rct1)
        kernel_fr2 = np.load(mpk_rct2)
        return kernel_fr1, kernel_fr2
    
    try:
        def submatch_and_mapping(cand, hatoms_restrict, args):
            if cand.empty: return cand
            cand['washed_mapped_smiles'] = cand['washed_isomeric_kekule_smiles'].swifter.apply(
                center.SetReactionCenter,args=args)
            return cand.dropna().query('@hatoms_restrict[0] <= heavy_atom_count <= @hatoms_restrict[1]')
        
        if not(precalc):
            with timer(process_name='Substructure matching'):
                cands = pd.read_table(cand_path,header=0,index_col=0, chunksize=chunk)
                plist_fr1 = []
                plist_fr2 = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, cand in enumerate(cands):
                        with timer(process_name=f'Process {i}'):
                            cand_fr1 = submatch_and_mapping(cand, rct1_hatoms_range, (1,True,))
                            p_fr1 = f'{tmpdir}/fr1_{i}.tsv'
                            cand_fr1.to_csv(p_fr1,sep='\t')
                            plist_fr1.append(p_fr1)

                            cand_fr2 = submatch_and_mapping(cand, rct2_hatoms_range, (2,True,))
                            p_fr2 = f'{tmpdir}/fr2_{i}.tsv'
                            cand_fr2.to_csv(p_fr2,sep='\t')
                            plist_fr2.append(p_fr2)

                    md.mk_dir(dir_cand)
                    tsv_merge(plist_fr1,merge_file=mp_rct1)
                    tsv_merge(plist_fr2,merge_file=mp_rct2)

        def SmilesExtractor(tpath,smiles_col,opath,downsize):
            if downsize is not None:
                size = -1
                with open(tpath,"r") as f:
                    for _ in f.readlines():
                        size += 1
                if size > downsize:
                    random_dice = np.array([1] * downsize + [0] * (size - downsize))
                else:
                    random_dice = np.array([1] * size)
                random_dice = random_dice.astype(bool)
                rng.shuffle(random_dice)
                endpoint = 0
            df = pd.read_table(tpath,header=0,index_col=0,chunksize=chunk)
            with open(opath,'w') as of:
                df_chunked = next(df)
                if downsize is not None:
                    chsize = df_chunked.shape[0]
                    df_chunked = df_chunked[random_dice[endpoint:endpoint+chsize]]
                    endpoint += chsize
                if not df_chunked.empty:
                    idx_list = df_chunked['EMOL_VERSION_ID'].to_list()
                    smi_list = df_chunked[smiles_col].to_list()
                    joined   = [f'{smi}\t{idx}' for idx, smi in zip(idx_list,smi_list)]
                    joined.append('')
                    smi_str  = '\n'.join(joined)
                    of.write(smi_str)
            with open(opath,'a') as of:
                for df_chunked in df:
                    if downsize is not None:
                        chsize = df_chunked.shape[0]
                        df_chunked = df_chunked[random_dice[endpoint:endpoint+chsize]]
                        endpoint += chsize
                    if not df_chunked.empty:
                        idx_list = df_chunked['EMOL_VERSION_ID'].to_list()
                        smi_list = df_chunked[smiles_col].to_list()
                        joined   = [f'{smi}\t{idx}' for idx, smi in zip(idx_list,smi_list)]
                        joined.append('')
                        smi_str  = '\n'.join(joined)
                        of.write(smi_str)
        
        opath_rct1 = f'{dir_cand}/{file_uni_name}_{reaction}_rct1_candidates_smiles.smi'
        opath_rct2 = f'{dir_cand}/{file_uni_name}_{reaction}_rct2_candidates_smiles.smi'

        SmilesExtractor(mp_rct1,'washed_isomeric_kekule_smiles',opath_rct1,downsize)
        SmilesExtractor(mp_rct2,'washed_isomeric_kekule_smiles',opath_rct2,downsize)

        ts_dict = {
            "reagent_file_list": [
                opath_rct1,
                opath_rct2
            ],
            "reaction_smarts": rc.template,
            "num_warmup_trials": 3,
            "num_ts_iterations": args['n_samples'],
            "evaluator_class_name": "ObjectiveEvaluatorByTanimotoKernel",
            "evaluator_arg": {
                "mod_path": model_path,
                "reaction_metaname": reaction
            },
            "ts_mode": "maximize",
            "log_filename": f"{dir_cand}/ts_logs.txt",
            "results_filename": f"{dir_cand}/ts_results.csv"
                }
        
        with open(f'{dir_cand}/ts_envs.json','w') as f:
            json.dump(ts_dict,f)

        run('TS_main_20240607.ts_main',
            f'{dir_cand}/ts_envs.json')
        
        lgr = get_logger(__name__,filename=ts_dict['log_filename'])
        lgr.info('Start validation of extracted compounds !')
        start = time()
        with tempfile.TemporaryDirectory() as tmpdir:
            dfs = pd.read_csv(f'{dir_cand}/ts_results.csv',index_col=None,header=0,chunksize=chunk)
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
                valid_list.append(f'{tmpdir}/{i}.tsv')
            tsv_merge(valid_list,f'{dir_cand}/ts_results_valid.tsv')
        
    except Exception as e:
        print(e)
        print('Reactant screening skip !')
        continue
          