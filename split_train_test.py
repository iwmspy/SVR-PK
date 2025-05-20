import os,json,sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)

import pandas as pd
import argparse, json

from models._CustomCVsplit import *
from utils.utility import AttrJudge


parser = argparse.ArgumentParser(description='Prepare dataset for modeling...')
parser.add_argument('-c', '--config', default=f'{pwd}/config/chembl_config_lv1.json', help='Configration')

args = parser.parse_args()

def validation_labels(tr_data, id_col):
    # val (For only MolCLR)
    grouped = tr_data.groupby('Rep_reaction')
    vals = []
    for _, group in grouped:
        unique_products = group.drop_duplicates('Product_raw')
        _, val_ids = train_test_split(unique_products[id_col], test_size=0.1, random_state=0)
        vals.extend(list(val_ids))
    return vals

def main():
    with open(args.config,'r') as f:
        confs = json.load(f)

    files = confs['files']
    index_col = confs['index_col']
    out_dir = AttrJudge(confs, 'out_dir', f'{pwd}/outputs')

    for file in files:
        file_uni_name = os.path.split(file)[-1].rsplit('.',1)[0]
        dir_preprocess = f'{out_dir}/preprocessed/{file_uni_name}'
        prep_data = pd.read_table(f'{dir_preprocess}/retro_{file_uni_name}_preprocessed.tsv',
            header = 0, index_col = 0)
        prep_data.index.name = 'original_id'
        prep_data = prep_data.reset_index()
        assert(prep_data.drop_duplicates(subset=['original_id']).shape[0] == prep_data.shape[0]), 'Duplicated original_id found!'

        # Perform both splits and store results in separate variables
        split1_result = CustomDissimilarRandomSplit(
            prep_data.copy(), index_col, 'Rep_reaction', 1, 'Product_raw')
        split2_result = CustomFragmentSpaceSplitbyFreq(
            prep_data.copy(), index_col, 'Precursors', 0.4, 'Rep_reaction')

        tr_data_split1, _, ts_data_split1, _ = split1_result
        tr_data_split2, _, ts_data_split2, _, _ = split2_result

        # Save original_id columns as newline-separated text files
        tr_data_split1['original_id'].to_csv(f'{dir_preprocess}/product-based_train_ids.txt', index=False, header=False)
        ts_data_split1['original_id'].to_csv(f'{dir_preprocess}/product-based_test_ids.txt', index=False, header=False)
        tr_data_split2['original_id'].to_csv(f'{dir_preprocess}/reactant-based_train_ids.txt', index=False, header=False)
        ts_data_split2['original_id'].to_csv(f'{dir_preprocess}/reactant-based_test_ids.txt', index=False, header=False)

        val_split1 = validation_labels(tr_data_split1, 'original_id')
        val_split2 = validation_labels(tr_data_split2, 'original_id')
        pd.Series(val_split1).to_csv(f'{dir_preprocess}/product-based_val_ids.txt', index=False, header=False)
        pd.Series(val_split2).to_csv(f'{dir_preprocess}/reactant-based_val_ids.txt', index=False, header=False)

if __name__=='__main__': 
    main()