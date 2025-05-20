## Modeling by splitted fingerprints

import os,json,sys,re
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)
import traceback

import pandas as pd
import numpy as np
import argparse, json, pickle

from models._CustomCVsplit import *
from models.modeling import ReactionGroupWrapperModeling
from utils.utility import timer,logger,AttrJudge,MakeDirIfNotExisting
from utils.chemutils import TransReactantByTemplate


parser = argparse.ArgumentParser(description='Modeling from retrosynthesized pairs...')
parser.add_argument('-c', '--config', default=f'{pwd}/config/chembl_config_lv1.json', help='Configration')

args = parser.parse_args()

mls_prd  = ['svr_tanimoto']
mls_rct  = ['svr_tanimoto_split','svr_tanimoto_average','svr_tanimoto']

def main():
    with open(args.config,'r') as f:
        confs = json.load(f)

    files = confs['files']
    index_col = confs['index_col']
    objective_col = confs['objective_col']
    split_level  = confs['split_level']
    augmentation = confs['augmentation'] if 'augmentation' in confs else False
    out_dir = AttrJudge(confs, 'out_dir', f'{pwd}/outputs')
    assert(isinstance(split_level,int) and split_level >= 0)

    dir_pred_wrap = f'{out_dir}/prediction_level{split_level}'
    if augmentation:
        dir_pred_wrap = f'{dir_pred_wrap}_augmented'
    MakeDirIfNotExisting(dir_pred_wrap)
    lgr = logger(filename=f'{out_dir}/logs/prediction_log.txt')
    lgr.write(f'Start {__file__}')
    lgr.write(f'Files : ---')
    for file in files:
        lgr.write(file)
    lgr.write(f'-----------')
    lgr.write(f'Index : {index_col}')
    lgr.write(f'Objective : {objective_col}')
    lgr.write(f'Split level : {split_level}')
    lgr.write(f'Augmentation : {augmentation}')
    lgr.write(f'Output : {out_dir}')

    for file in files:
        try:
            file_uni_name = os.path.split(file)[-1].rsplit('.',1)[0]
            dir_pred = f'{dir_pred_wrap}/{file_uni_name}'
            dir_preprocess = f'{out_dir}/preprocessed/{file_uni_name}'
            rxnmlmod = ReactionGroupWrapperModeling(mls_prd,mls_rct,outdir=dir_pred)
            with timer(process_name=f'Modeling of {file_uni_name}') as t:
                lgr.write(f'Start prediction of dataset {str(file)}')
                prep_data = pd.read_table(f'{dir_preprocess}/retro_{file_uni_name}_preprocessed.tsv',
                    header = 0, index_col = 0)
                if  split_level == 1:
                    with open(f'{dir_preprocess}/product-based_train_ids.txt', 'r') as f:
                        tr_indices = [int(line.strip()) for line in f if line.strip()]
                    with open(f'{dir_preprocess}/product-based_test_ids.txt', 'r') as f:
                        ts_indices = [int(line.strip()) for line in f if line.strip()]
                elif split_level == 2:
                    with open(f'{dir_preprocess}/reactant-based_train_ids.txt', 'r') as f:
                        tr_indices = [int(line.strip()) for line in f if line.strip()]
                    with open(f'{dir_preprocess}/reactant-based_test_ids.txt', 'r') as f:
                        ts_indices = [int(line.strip()) for line in f if line.strip()]

                tr_data = prep_data.loc[tr_indices]
                ts_data = prep_data.loc[ts_indices]

                if augmentation:
                    tr_data = TransReactantByTemplate(
                        tr_data, index_col, 'Product', 'Precursors', 'template', 
                        'Rep_reaction', objective_col, product_ECFP_col='Product_raw')
                tr_score_prd, tr_score_rct = rxnmlmod.run_cv(
                    tr_data,objective_col,'Product_raw','Precursors',index_col,'Rep_reaction',
                    df_whole=None,split_components=True)
                ts_score_prd, ts_score_rct = rxnmlmod.scoring(
                    ts_data,objective_col,'Product_raw','Precursors',index_col,'Rep_reaction',
                    df_whole=None,split_components=True)
                
                print('==Results of Training==')
                print('--Results of product modeling--')
                print(tr_score_prd)
                print('\n')
                print('--Results of reactant modeling--')
                print(tr_score_rct)
                print('\n')
                print('==Results of Test==')
                print('--Results of product modeling--')
                print(ts_score_prd)
                print('\n')
                print('--Results of reactant modeling--')
                print(ts_score_rct)
                print('\n')
                lgr.write(f'End prediction. Took {str(t.get_runtime())} sec.')

            with open(os.path.join(dir_pred,'mod.pickle'),'wb') as f:
                pickle.dump(rxnmlmod, f)
        
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
            lgr.write('Modeling skip!')

if __name__=='__main__': 
    main()