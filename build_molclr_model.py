## Modeling by splitted fingerprints

import os,json,sys,re
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)
import traceback
from tempfile import TemporaryDirectory

import pandas as pd
import numpy as np
import argparse, json
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

from models._CustomCVsplit import *
from models.molclr_interface import main as molclr_main
from utils.utility import timer,logger,AttrJudge,MakeDirIfNotExisting


parser = argparse.ArgumentParser(description='Construct MolCLR model using products...')
parser.add_argument('-c', '--config', default=f'{pwd}/config/chembl_config_lv1.json', help='Configration')

args = parser.parse_args()

def main():
    with open(args.config,'r') as f:
        confs = json.load(f)

    files = confs['files']
    index_col = confs['index_col']
    objective_col = confs['objective_col']
    split_level  = confs['split_level']
    out_dir = AttrJudge(confs, 'out_dir', f'{pwd}/outputs')
    assert(isinstance(split_level,int) and split_level >= 0)

    dir_pred_wrap = f'{out_dir}/prediction_level{split_level}'
    MakeDirIfNotExisting(dir_pred_wrap)
    lgr = logger(filename=f'{out_dir}/logs/prediction_level{split_level}_molclr_log.txt')
    lgr.write(f'Start {__file__}')
    lgr.write(f'Files : ---')
    for file in files:
        lgr.write(file)
    lgr.write(f'-----------')
    lgr.write(f'Index : {index_col}')
    lgr.write(f'Objective : {objective_col}')
    lgr.write(f'Split level : {split_level}')
    lgr.write(f'Output : {out_dir}')

    for file in files:
        try:
            file_uni_name = os.path.split(file)[-1].rsplit('.',1)[0]
            dir_pred = f'{dir_pred_wrap}/{file_uni_name}'
            dir_preprocess = f'{out_dir}/preprocessed/{file_uni_name}'
            with timer(process_name=f'Modeling of {file_uni_name}') as t:
                lgr.write(f'Start prediction of dataset {str(file)}')
                prep_data = pd.read_table(f'{dir_preprocess}/retro_{file_uni_name}_preprocessed.tsv',
                    header = 0, index_col = 0)
                if  split_level == 1:
                    with open(f'{dir_preprocess}/product-based_train_ids.txt', 'r') as f:
                        tr_indices = [int(line.strip()) for line in f if line.strip()]
                    with open(f'{dir_preprocess}/product-based_test_ids.txt', 'r') as f:
                        ts_indices = [int(line.strip()) for line in f if line.strip()]
                    with open(f'{dir_preprocess}/product-based_val_ids.txt', 'r') as f:
                        vl_indices = [int(line.strip()) for line in f if line.strip()]
                elif split_level == 2:
                    with open(f'{dir_preprocess}/reactant-based_train_ids.txt', 'r') as f:
                        tr_indices = [int(line.strip()) for line in f if line.strip()]
                    with open(f'{dir_preprocess}/reactant-based_test_ids.txt', 'r') as f:
                        ts_indices = [int(line.strip()) for line in f if line.strip()]
                    with open(f'{dir_preprocess}/reactant-based_val_ids.txt', 'r') as f:
                        vl_indices = [int(line.strip()) for line in f if line.strip()]
                    
                tr_data = prep_data.loc[tr_indices]
                ts_data = prep_data.loc[ts_indices]
                vl_data = prep_data.loc[vl_indices]

                # Remove rows from tr_data that have both index_col and 'Rep_reaction' present in vl_data
                vl_keys = set(zip(vl_data[index_col], vl_data['Rep_reaction']))
                tr_data = tr_data[~tr_data.apply(lambda row: (row[index_col], row['Rep_reaction']) in vl_keys, axis=1)]

                tr_data['split'] = 'train'
                ts_data['split'] = 'test'
                vl_data['split'] = 'val'

                final_data = pd.concat([tr_data, vl_data, ts_data])

                prd_results_dfs = []
                prd_scores_dfs  = []

                name_list = []
                metrics = ['train_r2', 'train_rmse', 'train_mae', 'val_r2', 'val_rmse', 'val_mae', 'test_r2', 'test_rmse', 'test_mae']

                with TemporaryDirectory(dir=dir_pred_wrap) as tmpdir:
                    for name, group in final_data.groupby('Rep_reaction'):
                        if group[group['split']=='test'].empty:
                            continue

                        name_list.append(name)

                        group = group.rename(columns={objective_col: 'obj'})
                        product_df = group.rename(columns={'Product_raw': 'smiles'})[[index_col, 'smiles', 'obj', 'split']]
                        product_df.drop_duplicates(subset=['smiles'], inplace=True)
                        
                        product_df.to_csv(f'{tmpdir}/{group["Rep_reaction"].values[0]}_product.csv', index=False)

                        args_prd = argparse.Namespace(
                            config_path=f'{pwd}/models/config_finetune.yaml',
                            data_path=f'{tmpdir}/{group["Rep_reaction"].values[0]}_product.csv',
                            model_name=f'{file_uni_name}_{name}'
                        )
                        product_results = molclr_main(args_prd)
                        product_results.index = product_df.index
                        prd_results_raw = pd.concat([product_df, product_results[['prediction','ground_truth']]],axis=1)
                        prd_results_raw['Rep_reaction'] = name
                        prd_results_dfs.append(prd_results_raw)
                        prd_scores_dfs.append([
                            r2_score(prd_results_raw[prd_results_raw['split']=='train']['obj'], prd_results_raw[prd_results_raw['split']=='train']['prediction']),
                            np.sqrt(mean_squared_error(prd_results_raw[prd_results_raw['split']=='train']['obj'], prd_results_raw[prd_results_raw['split']=='train']['prediction'])),
                            mean_absolute_error(prd_results_raw[prd_results_raw['split']=='train']['obj'], prd_results_raw[prd_results_raw['split']=='train']['prediction']),
                            r2_score(prd_results_raw[prd_results_raw['split']=='val']['obj'], prd_results_raw[prd_results_raw['split']=='val']['prediction']),
                            np.sqrt(mean_squared_error(prd_results_raw[prd_results_raw['split']=='val']['obj'], prd_results_raw[prd_results_raw['split']=='val']['prediction'])),
                            mean_absolute_error(prd_results_raw[prd_results_raw['split']=='val']['obj'], prd_results_raw[prd_results_raw['split']=='val']['prediction']),
                            r2_score(prd_results_raw[prd_results_raw['split']=='test']['obj'], prd_results_raw[prd_results_raw['split']=='test']['prediction']),
                            np.sqrt(mean_squared_error(prd_results_raw[prd_results_raw['split']=='test']['obj'], prd_results_raw[prd_results_raw['split']=='test']['prediction'])),
                            mean_absolute_error(prd_results_raw[prd_results_raw['split']=='test']['obj'], prd_results_raw[prd_results_raw['split']=='test']['prediction']),
                                     ])

                # Save results_dfs to dir_pred
                prd_results_df = pd.concat(prd_results_dfs, ignore_index=True)
                MakeDirIfNotExisting(dir_pred)
                prd_results_df.to_csv(f'{dir_pred}/prd_molclr_results.csv')

                # Save scores_dfs to dir_pred
                prd_scores_df = pd.DataFrame(prd_scores_dfs, columns=metrics, index=name_list)
                prd_scores_df.to_csv(f'{dir_pred}/prd_molclr_scores.csv')

                print('--Results of product modeling--')
                print(prd_scores_df)
                print('\n')
                lgr.write(f'End prediction. Took {str(t.get_runtime())} sec.')
        
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