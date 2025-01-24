''' Preprocess (Data preparation) for retrosynthesised datasets
    This code works following below flow.
    1. Classify the dataset by used reaction-template
    2. Do below processes for each classified datasets
        a. Delete only 1 fragment by a product
        b. Name the dataset by USPTO-registered name
'''

from copy import deepcopy
import argparse,os,json,sys

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)

import pandas as pd

from utils.utility import timer, MakeDirIfNotExisting, logger, AttrJudge
from curation.preprocess import DataPreprocessing, Summarizer


parser = argparse.ArgumentParser(description='Preprocess generated data...')
parser.add_argument('-c', '--config', default=f'{pwd}/config/example_config.json', help='Configration')
args = parser.parse_args()


def main():
    with open(args.config,'r') as f:
        confs = json.load(f)

    files = confs['files']
    index_col = confs['index_col']
    out_dir = AttrJudge(confs, 'out_dir', f'{pwd}/outputs')

    dir_dataset = f'{out_dir}/datasets'
    dir_prep    = f'{out_dir}/preprocessed'
    MakeDirIfNotExisting(dir_prep)
    lgr = logger(filename=f'{dir_prep}/preprocess_log.txt')
    lgr.write(f'Start {__file__}')
    
    for file in files:
        try:
            file_uni_name = os.path.split(file)[-1].rsplit('.',1)[0]
            dir_prep_file = f'{dir_prep}/{file_uni_name}'
            with timer(process_name=f'Preprocess of {file_uni_name}') as t:
                lgr.write(f'Start preprocessing of dataset {str(file)}')
                MakeDirIfNotExisting(dir_prep_file)

                data = pd.read_table(
                    f'{dir_dataset}/retrosynthesis_{file_uni_name}.tsv', 
                    index_col=0, header=0)
                data_preprocessed = DataPreprocessing(
                    data,index_col,'template','Product','Precursors','USPTO_id')
                
                df_template_uspto = Summarizer(data_preprocessed,index_col)
                df_template_uspto.to_csv(
                    f'{dir_prep}/retro_{file_uni_name}_calculate_summary.tsv',
                    sep='\t')
                data_preprocessed.to_csv(
                    f'{dir_prep_file}/retro_{file_uni_name}_preprocessed.tsv',
                    sep='\t')
                lgr.write(f'End preprocessing. Took {str(t.get_runtime())} sec.')
        except Exception as e:
            lgr.write(e)
            lgr.write('Preprocessing skip!')

if __name__=='__main__':
    main()