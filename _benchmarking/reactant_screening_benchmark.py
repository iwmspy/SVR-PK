''' Fragment screening
    1. Search substructure (reaction center) of fragment -> candidates
    2. Calculate similarity between candidates and trainings -> Filter by sim >= 0.6
    3. Combine fragment -> candpairs
    4. Calculate kernel val between candpairs and trainings -> Filter by val >= 0.6
'''

import os,json,sys
import pickle
import argparse
import tempfile

import pandas as pd
import numpy as np
from rdkit import Chem
from time import time
from tqdm import tqdm
from scipy.sparse import csr_matrix,vstack
import swifter
from joblib import Parallel, delayed, cpu_count

# To import modules we defined
pwd_deep = os.path.dirname(os.path.abspath(__file__))
pwd = os.path.abspath(os.path.join(pwd_deep,'..'))
os.chdir(pwd)
sys.path.append(pwd)

from models._kernel_and_mod import funcTanimotoSklearn
from models.modeling import *
from screening.screening_mod import ReactantScreening
from utils.utility import mean_of_top_n_elements, tsv_merge, mkdir, timer, AttrJudge, logger, MakeDirIfNotExisting, ArraySplitByN
from utils.chemutils import ReactionCenter, reactor, is_valid_molecule, MorganbitCalcAsVectorFromSmiles
# from utils.clustering import NeighborsSimilarityForSparseMat,NearestNeighborSearchFromSmiles

# Define constants
CHUNK      = 100000
CHUNK_RCT1 = 1000
CHUNK_RCT2 = 100
MAX_COMBS  = 1000000
R_OF_SCORE = 75
N_SAMPLES  = 1000000
N_SAMPLES_PER_RCT = int(np.sqrt(N_SAMPLES))
SEED = 0

parser = argparse.ArgumentParser(description='Collect reactants and combine them...')
parser.add_argument('-c', '--config', default=f'{pwd}/config/example_config.json',
    help='Configration')


args = parser.parse_args()


## Read configurations from json file
confs = json.load(open(args.config,'r'))
# Settings used in this program
files            = confs['files']
cand_path        = os.path.abspath(AttrJudge(confs, 'cand_path', f'{pwd}/emolecule/emolecule_compounds_curated.tsv'))
out_dir          = AttrJudge(confs, 'out_dir', f'{pwd_deep}/outputs') 
ext_ratio        = AttrJudge(confs, 'ext_ratio', 1e-5) 
reactions        = confs['reactions']
split_level      = confs['split_level']
precalc          = AttrJudge(confs, 'precalc', False)

out_dir_sc_base  = f'{out_dir}/reactant_combination_level{split_level}'
MakeDirIfNotExisting(out_dir_sc_base)

if isinstance(reactions, str):
    reactions = [reactions for _ in range(len(files))]
assert(len(files)==len(reactions))

def main():
    for odr, file in enumerate(files):
        file_uni_name = os.path.split(file)[-1].rsplit('.',1)[0]
        lgr = logger(filename=f'{out_dir_sc_base}/screening_log_{file_uni_name}.txt')
        reaction = reactions[odr]
        lgr.write(f'<<< Start reactant combination of {file_uni_name} >>>')

        rs = ReactantScreening(file_uni_name, reaction, lgr, confs)
        
        try:  
            if not precalc:
                cands = pd.read_table(cand_path,header=0,index_col=0, chunksize=CHUNK)
                arg_rct1 = (1,True,)
                arg_rct2 = (2,True,)
                rs.SubmatchAndMapping(cands, arg_rct1, arg_rct2)
                rs.CandsKernelGenerator()

            rs.SVRpredCombinations(ext_ratio=ext_ratio,njobs=os.cpu_count())
            # rs.SVRpredAnalysis(njobs=os.cpu_count())
            # rs.SVRpredAnalysis(njobs=1)
            rs.ResultsExtractor(njobs=os.cpu_count())
            # rs.ResultsExtractor(njobs=1)
        
        except Exception as e:
            lgr.write(e)
            lgr.write('Reactant screening skip !')
            continue

if __name__=='__main__':
    main()