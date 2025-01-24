## Generate dataset(s) using Retrosynthesis

import os,json,sys,argparse

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)

import numpy as np
import pandas as pd

from retrosynthesis import retrosep
from utils.utility import check_group_vals, MakeDirIfNotExisting, timer, logger, AttrJudge


parser = argparse.ArgumentParser(description='Retrosynthesize actual molecules...')
parser.add_argument('-c', '--config', default=f'{pwd}/config/example_config.json', help='Configration')

args = parser.parse_args()


def duplicate_check(df, idx_col, obj_col):
    # If same compounds (products) have different objective values, check if the difference is in threshold
    # If difference <= threshold, then each objectives are averaged and use this value as a new objective
    # If difference >= threshold, then data rows containing this product are disposed
    # This procedure will triggered if products don't have chiral centers, or some other reasons...
    df_copy = df.copy()
    df_group = df_copy.loc[:,[idx_col,obj_col]].groupby(idx_col)[obj_col].apply(check_group_vals)
    for p in df_group.index:
        if df_group[p]==1:
            df_copy = df_copy[df_copy[idx_col]!=p]
        elif df_group[p]==0:
            temp = df_copy[df_copy[idx_col]==p]
            maxsimrow = temp.head(1)
            del maxsimrow[obj_col]
            maxsimrow[obj_col] = np.mean(temp[obj_col])
            maxsimrow['dupl_id'] = ','.join(temp[idx_col])
            df_copy = df_copy[df_copy[idx_col]!=p]
            df_copy = pd.concat([df_copy,maxsimrow])
    return df_copy.set_index(idx_col)


def main():
    with open(args.config,'r') as f:
        confs = json.load(f)

    files = confs['files']
    index_col = confs['index_col']
    model_smiles_col = confs['model_smiles_col']
    objective_col = confs['objective_col']
    out_dir = AttrJudge(confs, 'out_dir', f'{pwd}/outputs')
    
    dir_dataset  = f'{out_dir}/datasets'
    MakeDirIfNotExisting(dir_dataset)
    lgr = logger(filename=f'{dir_dataset}/generate_log.txt')
    lgr.write(f'Start {__file__}')
    
    lgr.write('Loaded configs.')
    lgr.write(f'files={str(files)}')
    lgr.write(f'index_col={str(index_col)}')
    lgr.write(f'model_smiles_col={str(model_smiles_col)}')
    lgr.write(f'objective_col={str(objective_col)}')

    for file in files:
        file_uni_name = os.path.split(file)[-1].rsplit('.',1)[0]
        with timer(f'Retrosynthesis of {file_uni_name}') as t:
            lgr.write(f'Start retrosynthesis of dataset {str(file)}')
            data     = pd.read_table(file, header=0)
            data_uni = duplicate_check(data, index_col, objective_col)
            lgr.write(f'Dataset size: {str(data.shape[0])}')
            lgr.write(f'Dataset size (unique): {str(data_uni.shape[0])}')
            smi_list = data_uni.loc[:, model_smiles_col].to_list()
            idx_list = data_uni.index.to_list()
            obj_list = data_uni.iloc[:,data_uni.columns.get_loc(objective_col)]
            ext_list = []
            ext_idx  = []
            columns  = [data_uni.index.name,'Product','Precursors','template', 
                'USPTO_id', 'class','prod_smiles','prod_sim',
                'prec_smiles','prec_sim','overall_sim',objective_col]
            syn_df = pd.DataFrame(columns=columns)

            for idx, i in enumerate(smi_list):
                try:
                    # Do Retrosynthesis
                    syn = retrosep.do_one(i,rc_cumurate=False)
                    for syn_ in syn:
                        syn_.insert(0,idx_list[idx])
                        syn_.append(obj_list[idx])
                    sd  = pd.DataFrame(syn,columns=columns).sort_values(
                        'overall_sim',ascending=False,kind='mergesort')
                    if sd.shape[0]<=10:
                        syn_df_tmp = sd
                    else:
                        cnt = 10
                        loc = sd.columns.get_loc('overall_sim')
                        while sd.iloc[cnt-1,loc]==sd.iloc[cnt,loc]:
                            cnt += 1
                            if sd.iloc[cnt,loc]==sd.iloc[-1,loc]:
                                break
                        syn_df_tmp = sd.head(cnt)
                    syn_df = pd.concat([syn_df, syn_df_tmp])
                except:
                    ext_list.append(i)
                    ext_idx.append(idx)

            syn_df = syn_df.reset_index(drop=True)

            assert not(syn_df.empty), 'No compounds have retrosynthesized !'
            assert syn_df[syn_df.duplicated(subset=['Precursors'])].shape[0]==0, \
                'Something went wrong during retrosynthesis !'

            retrosep.analysis_of_retrosynthesis(syn_df, 
                name=f'{dir_dataset}/retrosynthesis_{file_uni_name}_summary.txt')
            ext_df = pd.DataFrame(ext_list, index = data_uni.index[ext_idx], columns = ['Products'])

            syn_df.to_csv(f'{dir_dataset}/retrosynthesis_{file_uni_name}.tsv', sep='\t')
            ext_df.to_csv(f'{dir_dataset}/retrosynthesis_{file_uni_name}_error.tsv',sep='\t')
            lgr.write(f'End retrosynthesis. Took {str(t.get_runtime())} sec.')
            lgr.write(f'Eliminated {str(ext_df.shape[0])} compounds by this procedure.')


if __name__=='__main__':
    main()