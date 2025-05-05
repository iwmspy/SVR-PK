''' Fragment screening
    1. Search substructure (reaction center) of fragment -> candidates
    2. Calculate similarity between candidates and trainings -> Filter by sim >= 0.6
    3. Combine fragment -> candpairs
    4. Calculate kernel val between candpairs and trainings -> Filter by val >= 0.6
'''

import argparse
import os,json,re,sys,traceback

import pandas as pd
import numpy as np

# To import modules we defined
pwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(pwd)
sys.path.append(pwd)

from models.modeling import *
from screening.screening_mod import ReactantScreening
from utils.utility import AttrJudge, logger, MakeDirIfNotExisting

# Define constants
CHUNK      = 100000
CHUNK_RCT1 = 1000
CHUNK_RCT2 = 100
R_OF_SCORE = 75
N_SAMPLES  = 1000000
N_SAMPLES_PER_RCT = int(np.sqrt(N_SAMPLES))
SEED = 0

parser = argparse.ArgumentParser(description='Collect reactants and combine them...')
parser.add_argument('-c', '--config', default=f'{pwd}/config/chembl_config_for_screening_10k.json',
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
ext_ratio        = AttrJudge(confs, 'ext_ratio', 1e-5) 
reactions        = confs['reactions']
split_level      = confs['split_level']
n_samples        = confs['n_samples']
n_jobs           = AttrJudge(confs, 'n_jobs', os.cpu_count())
augmentation     = AttrJudge(confs, 'augmentation', False)
downsize         = AttrJudge(confs, 'downsize_sc', None)
precalc          = AttrJudge(confs, 'precalc', False)
postpro          = AttrJudge(confs, 'postprocess', False)

if augmentation:
    odir_pref = f'{out_dir}/reactant_combination_level{split_level}_augmented'
else:
    odir_pref = f'{out_dir}/reactant_combination_level{split_level}'

if downsize is not None:
    out_dir_sc_base  = f'{odir_pref}_{n_samples}_rc{downsize}'
else:
    out_dir_sc_base  = f'{odir_pref}_{n_samples}'
MakeDirIfNotExisting(out_dir_sc_base)

if isinstance(reactions, str):
    reactions = [reactions for _ in range(len(files))]
assert(len(files)==len(reactions))

def main():
    rdicts = []
    for odr, file in enumerate(files):
        file_uni_name = os.path.split(file)[-1].rsplit('.',1)[0]
        lgr = logger(filename=f'{out_dir}/logs/screening_log_{file_uni_name}.txt')
        reaction = reactions[odr]
        lgr.write(f'<<< Start reactant combination of {file_uni_name} >>>')
        lgr.write('======Config sets======')
        lgr.write(f'file: {file}')
        lgr.write(f'reaction: {reaction}')
        [lgr.write(f'{key}: {val}') for key, val in confs.items() if key not in ['files','reactions']]
        lgr.write('=======================')

        rs = ReactantScreening(file_uni_name, reaction, lgr, confs)
        
        try:  
            if not precalc:
                cands = pd.read_table(cand_path,header=0,index_col=0, chunksize=CHUNK)
                arg_rct1 = (1,True,)
                arg_rct2 = (2,True,)
                rs.SubmatchAndMapping(cands, arg_rct1, arg_rct2)
            rs.CandsKernelGenerator()
            rs.SVRpredCombinations(ext_ratio=ext_ratio,njobs=n_jobs)
            # rs.SVRpredAnalysis(njobs=os.cpu_count())
            rs.ResultsExtractor(njobs=n_jobs)
            rs.PredictFromProducts(njobs=n_jobs,retrieve_size=n_samples)
            rs.RouteExaminator(retrieved=True)
            rdicts.append(rs.res_dict.copy())
        
        except Exception as e:
            error_class = type(e)
            error_description = str(e)
            err_msg = '%s: %s' % (error_class, error_description)
            lgr.write(err_msg)
            tb = traceback.extract_tb(sys.exc_info()[2])
            trace = traceback.format_list(tb)
            lgr.write('---- traceback ----')
            for line in trace:
                if '~^~' in line:
                    lgr.write(line.rstrip())
                else:
                    text = re.sub(r'\n\s*', ' ', line.rstrip())
                    lgr.write(text)
            lgr.write('-------------------')
            continue
    rdf = pd.DataFrame(rdicts)
    rdf.to_csv(f'{out_dir_sc_base}/svr-pk_screening_result_analysis.tsv',sep='\t')

if __name__=='__main__':
    main()