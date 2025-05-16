## Modeling by splitted fingerprints

import os,json,sys,re
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)
import traceback
from tempfile import TemporaryDirectory

import pandas as pd
import numpy as np
import argparse, json, pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

from models._CustomCVsplit import *
from models.molclr_interface import main as molclr_main
from utils.utility import timer,logger,AttrJudge,MakeDirIfNotExisting
from utils.chemutils import TransReactantByTemplate


parser = argparse.ArgumentParser(description='Retrosynthesize actual molecules...')
parser.add_argument('-c', '--config', default=f'{pwd}/config/chembl_config_lv1.json', help='Configration')
# parser.add_argument('-c', '--config', default=f'{pwd}/config/chembl_config_lv2.json', help='Configration')

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
    lgr = logger(filename=f'{out_dir}/logs/prediction_level{split_level}_molclr_log.txt')
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
            with timer(process_name=f'Modeling of {file_uni_name}') as t:
                lgr.write(f'Start prediction of dataset {str(file)}')
                prep_data = pd.read_table(f'{dir_preprocess}/retro_{file_uni_name}_preprocessed.tsv',
                    header = 0, index_col = 0)
                if  split_level == 1:
                    tr_data, tr_whole_data, ts_data, ts_whole_data    = CustomDissimilarRandomSplit(
                        prep_data,index_col,'Rep_reaction',split_level,'Product_raw')
                elif split_level == 2:
                    tr_data, tr_whole_data, ts_data, ts_whole_data, _ = CustomFragmentSpaceSplitbyFreq(
                        prep_data,index_col,'Precursors',0.4,'Rep_reaction')
                if augmentation:
                    tr_data = TransReactantByTemplate(
                        tr_data, index_col, 'Product', 'Precursors', 'template', 
                        'Rep_reaction', objective_col, product_ECFP_col='Product_raw')
                
                tr_data['split'] = 'train'
                ts_data['split'] = 'test'

                final_data = pd.concat([tr_data, ts_data])

                prd_results_dfs = []
                prd_scores_dfs  = []
                # rct_results_dfs = []
                # rct_scores_dfs  = []

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
                        train_group, val_group = train_test_split(product_df[product_df['split'] == 'train'], test_size=0.1, random_state=0)
                        product_df.loc[val_group.index, 'split'] = 'val'
                        # reactant_df = group.rename(columns={'Precursors': 'smiles'})[[index_col, 'smiles', 'obj', 'split']]
                        # train_group, val_group = train_test_split(reactant_df[reactant_df['split'] == 'train'], test_size=0.1, random_state=0)
                        # reactant_df.loc[val_group.index, 'split'] = 'val'
                        
                        product_df.to_csv(f'{tmpdir}/{group["Rep_reaction"].values[0]}_product.csv', index=False)
                        # reactant_df.to_csv(f'{tmpdir}/{group["Rep_reaction"].values[0]}_reactant.csv', index=False)

                        args_prd = argparse.Namespace(
                            config_path=f'{pwd}/models/config_finetune.yaml',
                            data_path=f'{tmpdir}/{group["Rep_reaction"].values[0]}_product.csv',
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

                        # args_rct = argparse.Namespace(
                        #     config_path=f'{pwd}/models/config_finetune.yaml',
                        #     data_path=f'{tmpdir}/{group["Rep_reaction"].values[0]}_reactant.csv',
                        # )
                        # reactant_results = molclr_main(args_rct)
                        # rct_results_raw = pd.concat([reactant_df, reactant_results],axis=1,ignore_index=True)
                        # rct_results_raw['Rep_reaction'] = name
                        # rct_results_dfs.append(rct_results_raw)

                        # Group rct_results_raw by index_col and calculate the mean for numeric columns
                        # For non-numeric columns, take the first entry in each group
                        # rct_results_grouped = rct_results_raw.groupby(index_col).agg(
                        #     lambda x: x.mean() if np.issubdtype(x.dtype, np.number) else x.iloc[0]
                        # ).reset_index()

                        # rct_scores_dfs.append([
                        #     r2_score(rct_results_grouped[rct_results_grouped['split']=='train']['obj'], rct_results_grouped[rct_results_grouped['split']=='train']['prediction']),
                        #     np.sqrt(mean_squared_error(rct_results_grouped[rct_results_grouped['split']=='train']['obj'], rct_results_grouped[rct_results_grouped['split']=='train']['prediction'])),
                        #     mean_absolute_error(rct_results_grouped[rct_results_grouped['split']=='train']['obj'], rct_results_grouped[rct_results_grouped['split']=='train']['prediction']),
                        #     r2_score(rct_results_grouped[rct_results_grouped['split']=='val']['obj'], rct_results_grouped[rct_results_grouped['split']=='val']['prediction']),
                        #     np.sqrt(mean_squared_error(rct_results_grouped[rct_results_grouped['split']=='val']['obj'], rct_results_grouped[rct_results_grouped['split']=='val']['prediction'])),
                        #     mean_absolute_error(rct_results_grouped[rct_results_grouped['split']=='val']['obj'], rct_results_grouped[rct_results_grouped['split']=='val']['prediction']),
                        #     r2_score(rct_results_grouped[rct_results_grouped['split']=='test']['obj'], rct_results_grouped[rct_results_grouped['split']=='test']['prediction']),
                        #     np.sqrt(mean_squared_error(rct_results_grouped[rct_results_grouped['split']=='test']['obj'], rct_results_grouped[rct_results_grouped['split']=='test']['prediction'])),
                        #     mean_absolute_error(rct_results_grouped[rct_results_grouped['split']=='test']['obj'], rct_results_grouped[rct_results_grouped['split']=='test']['prediction']),
                        #              ])
                # Save results_dfs to dir_pred
                prd_results_df = pd.concat(prd_results_dfs, ignore_index=True)
                # rct_results_df = pd.concat(rct_results_dfs, ignore_index=True)
                MakeDirIfNotExisting(dir_pred)
                prd_results_df.to_csv(f'{dir_pred}/prd_molclr_results.csv', index=False)
                # rct_results_df.to_csv(f'{dir_pred}/rct_molclr_results.csv', index=False)

                # Save scores_dfs to dir_pred
                prd_scores_df = pd.DataFrame(prd_scores_dfs, columns=metrics, index=name_list)
                # rct_scores_df = pd.DataFrame(rct_scores_dfs, columns=metrics, index=name_list)
                prd_scores_df.to_csv(f'{dir_pred}/prd_molclr_scores.csv')
                # rct_scores_df.to_csv(f'{dir_pred}/rct_molclr_scores.csv')

                print('--Results of product modeling--')
                print(prd_scores_df)
                print('\n')
                # print('--Results of reactant modeling--')
                # print(rct_scores_df)
                # print('\n')
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