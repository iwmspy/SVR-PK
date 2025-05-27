import argparse,os,sys
from utils.utility import run

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)

parser = argparse.ArgumentParser(description='Run the SVR-PK pipeline for retrosynthesis tasks.')
parser.add_argument('-c', '--config', default=f'{pwd}/config/chembl_config_lv1_augment.json', help='Configration')
args = parser.parse_args()

if __name__=='__main__':
	run('generate_reactant_pairs')
	run('preprocess')
	run('split_train_test')
	run('build_model')
	