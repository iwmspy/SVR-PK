## Modeling by splitted fingerprints

import argparse,os,sys


pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)

parser = argparse.ArgumentParser(description='Retrosynthesize actual molecules...')
parser.add_argument('-c', '--config', default=f'{pwd}/config/chembl_config_lv1_augment.json', help='Configration')
args = parser.parse_args()


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


def run(app):
    app_cls=importstr(app)
    app_cls.args.config = args.config
    app_cls.main()


if __name__=='__main__':
	run('generate_dataset_retrosynthesis')
	run('preprocess_for_retrosynthesis')
	run('modeling_retrosynthesis_ecfp_split')
	