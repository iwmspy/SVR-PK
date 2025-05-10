''' Fragment screening
    1. Search substructure (reaction center) of fragment -> candidates
    2. Calculate similarity between candidates and trainings -> Filter by sim >= 0.6
    3. Combine fragment -> candpairs
    4. Calculate kernel val between candpairs and trainings -> Filter by val >= 0.6
'''

import os
import pickle
import tempfile

import pandas as pd
import numpy as np
from rdkit import Chem
from time import time
from tqdm import tqdm
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed, cpu_count

from models._kernel_and_mod import funcTanimotoSklearn
from models.modeling import *
from utils.utility import tsv_merge, timer, logger, MakeDirIfNotExisting, ArraySplitByN
from utils.chemutils import ReactionCenter, reactor
from utils.analysis import ValidityAndSuggestedRouteExaminator

# Define constants
CHUNK = 100000
SEED  = 0

rng = np.random.default_rng(seed = SEED)

def _mod_svrpred(kmat1, kmat2, coeff, bias, worker_id, thres_score=None, logger=None):
    wp    = logger if logger is not None else print
    nfr1  = len(kmat1)
    fr1   = np.arange(nfr1)
    nfr2  = len(kmat2)
    combs = list()
    for i in range(nfr2):
        if worker_id == 0: # monitoring progress
            wp(f'Processing batch {i}/{nfr2}')
        fr2   = np.full(nfr1,i)
        score = (kmat1 * kmat2[i,:] * coeff).sum(axis=1) + bias
        if thres_score is not None:
            score_indices = np.where(score >= thres_score)[0]
            if not len(score_indices):
                continue
            combs_tmp = np.stack([fr1[score_indices], fr2[score_indices], score[score_indices]])
        else:
            combs_tmp = np.stack([fr1, fr2, score])
        combs.append(combs_tmp.T)
    return combs

def _mod_svrpred_extract(kmat1, kmat2, coeff, bias, worker_id, thres_score=None, save_name=None, logger=None):
    combs = _mod_svrpred(kmat1, kmat2, coeff, bias, worker_id, thres_score, logger)
    if len(combs):
        combs_array = np.concatenate(combs,axis=0)
        if save_name is None:
            return combs_array
        np.savetxt(save_name, combs_array, delimiter='\t')
        return save_name
    else:
        return None

def _mod_svrpred_analysis(kmat1, kmat2, coeff, bias, worker_id, ext_ratios_dict=None):
    d_shape = {}
    min_thres = min([d['threshold'] for d in ext_ratios_dict.values()]) if isinstance(ext_ratios_dict, dict) and not(None in ext_ratios_dict) \
        else None
    combs = _mod_svrpred(kmat1, kmat2, coeff, bias, worker_id, min_thres)
    combs_array = np.concatenate(combs,axis=0)
    score_ravel = combs_array[:, -1].ravel()
    for ratio, d in ext_ratios_dict.items():
        if ratio is not None:
            d_shape[ratio] = len(np.where(score_ravel >= d['threshold'])[0])
        else:
            d_shape[ratio] = len(score_ravel)
    return d_shape

def EachArrayLengthCalclator(arrays):
        mcumurateshape = 0
        mcumurateshape_array = np.zeros((len(arrays)+1),dtype=int)
        for i,mat in enumerate(arrays):
            mcumurateshape += mat.shape[0]
            mcumurateshape_array[i+1] = mcumurateshape
        return mcumurateshape_array

def extract_n_samples_by_threshold(tsv_path: str, retrieve_size=1000000):
    samples = np.loadtxt(tsv_path,delimiter='\t',dtype=np.float32)
    if samples.shape[0] <= retrieve_size:
        return None, samples[:,:-1], samples[:,-1]
    candi_splits = np.linspace(min(samples[:,-1]), max(samples[:,-1]), 100)
    count_over_thres = np.array([np.sum(samples[:,-1] > thres) for thres in candi_splits])
    greater_equal_indices = np.where(count_over_thres >= retrieve_size)
    greater_equal_values = count_over_thres[greater_equal_indices]
    min_distance = np.argmin(greater_equal_values)
    min_indice   = greater_equal_indices[0][min_distance]
    samples_ret  = samples[samples[:,-1] >= candi_splits[min_indice]]
    return candi_splits[min_indice], samples_ret[:,:-1], samples_ret[:,-1]


def NeatCombinationExtractor(array_1: np.array, array_2: np.array, n_samples: int):
    n_samples_per_rct = int(np.sqrt(n_samples))

    nfr1  = len(array_1)
    nfr2  = len(array_2)

    if nfr2 < n_samples_per_rct and nfr1 > int(n_samples / nfr2):
        array_1_sample = rng.choice(array_1,size=int(n_samples / nfr2),replace=False,shuffle=False)
    elif nfr2 < n_samples_per_rct and nfr1 <= int(n_samples / nfr2):
        array_1_sample = array_1.copy()
    else: 
        array_1_sample = rng.choice(array_1,size=n_samples_per_rct,replace=False,shuffle=False) \
            if nfr1 > n_samples_per_rct else array_1.copy()

    if nfr1 < n_samples_per_rct and nfr2 > int(n_samples / nfr1):
        array_2_sample = rng.choice(array_2,size=int(n_samples / nfr1),replace=False,shuffle=False)
    elif nfr1 < n_samples_per_rct and nfr2 <= int(n_samples / nfr1):
        array_2_sample = array_2.copy()
    else: 
        array_2_sample = rng.choice(array_2,size=n_samples_per_rct,replace=False,shuffle=False) \
            if nfr2 > n_samples_per_rct else array_2.copy()
    
    return array_1_sample, array_2_sample


def close_logger(logger):
    """Close all handlers associated with the logger."""
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

def ShapeConcatenator(ret: list, shape: np.array, read_f=False):
    ret_with_shape = list()
    for r,s in zip(ret,shape):
        if r is not None:
            if read_f:
                r = np.loadtxt(r, delimiter="\t")
            if r.ndim == 1:
                nrow = r.shape[0]
                r = r.reshape((-1,nrow))
            r_w_shape = r.copy()
            r_w_shape[:,1] += s
            ret_with_shape.append(r_w_shape)
    return ret_with_shape


class ReactantScreening:
    def __init__(self,file_uni_name,reaction,logger=None, args=dict()) -> None:
        self.args = args
        self.LoadMod(file_uni_name)
        self.reaction = reaction if reaction!='best' \
            else self.rxnmlmod.best_subgroup_estimators_index_['svr_tanimoto_split'][0]
        self.LoadKernelsAndComparingDatas(file_uni_name,self.reaction)
        self.ConstantsExtractor()
        self.logger = logger
        self.lgr = lambda x: logger.write(x) if logger is not None else print(x)
        self.lgr(f'Load mod {self.model_path}')
        self.lgr(f'Load train {self.path_rct_train}, {self.path_prd_train}')
        self.lgr(f'Load test {self.path_rct_test}')
        self.res_dict = {}
        self.res_dict['dataset'] = f'{file_uni_name}.{reaction}'
    
    def _mod_submatch_and_mapping(self, cand, hatoms_restrict, args, path, unique_reagent_set):
        if cand.empty: return cand
        cand['mapped_smiles'] = cand[self.args['cand_smiles_col']].swifter.apply(
            self.center.SetReactionCenter,args=args)
        cand_matched = cand.dropna().query('@hatoms_restrict[0] <= heavy_atom_count <= @hatoms_restrict[1]').drop_duplicates('mapped_smiles')
        cand_matched_unique = cand_matched[~cand_matched['mapped_smiles'].isin(unique_reagent_set)]
        cand_matched_unique.to_csv(path, sep='\t')
        [unique_reagent_set.add(smi) for smi in cand['mapped_smiles']]
    
    def _mod_cands_knl_gen(self, cands_chunked, train_bits, path_dict, pool, njobs):
        def _kernel_gen_with_parallel(bits: csr_matrix, train_bits, njobs):
            n_jobs  = njobs if njobs > 0 else os.cpu_count() + njobs + 1
            bits_sp = np.array_split(bits.toarray(), n_jobs)
            knls_sp = Parallel(n_jobs=n_jobs, backend='threading')(
                [delayed(funcTanimotoSklearn)(csr_matrix(bit),train_bits) for bit in bits_sp]
            )
            return np.concatenate(knls_sp,0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sh_rct, sh_dup_rct = 0, 0
            pdict_cands = {
                'metadata' : []
                }
            for i, cands_meta in enumerate(cands_chunked):
                cands_bits_spmat = self.rxnmlmod._var_gen(cands_meta,'mapped_smiles',False)
                sim_mat_array    = _kernel_gen_with_parallel(cands_bits_spmat, train_bits, njobs)
                
                cands_meta['max_kernel_val'] = np.max(sim_mat_array,axis=1)
                sh_rct += cands_meta.shape[0]
                sh_dup_rct += cands_meta[cands_meta['mapped_smiles'].apply(pool)].shape[0]

                p_cands_pref  = f'{tmpdir}/fr_{i}_cands'
                cands_meta.to_csv(f'{p_cands_pref}_metadata.tsv',sep='\t')
                pdict_cands['metadata'] = pdict_cands['metadata'] + [f'{p_cands_pref}_metadata.tsv']
                sim_mat_array_whole = np.concatenate([sim_mat_array_whole, sim_mat_array]) \
                    if i!=0 else sim_mat_array.copy()
            tsv_merge(pdict_cands['metadata'], path_dict['metadata'])
            np.save(path_dict['kernel'], sim_mat_array_whole)
        return sh_rct, sh_dup_rct
    
    def randomselector(self, path, index):
        chunked_df = pd.read_table(path,index_col=0,header=0,chunksize=CHUNK)
        size = 0
        for df in chunked_df:
            size += df.shape[0] 
        if size > self.args[index]:
            random_dice = np.array([1] * self.args[index] + [0] * (size - self.args[index]))
        else:
            random_dice = np.array([1] * size)
        random_dice = random_dice.astype(bool)
        rng.shuffle(random_dice)
        endpoint = 0
        cands = pd.read_table(path,index_col=0,header=0,chunksize=CHUNK)
        with tempfile.TemporaryDirectory() as tmpdir:
            sel_list = []
            for i,cand in enumerate(cands):
                chsize = cand.shape[0]
                cand_selected = cand[random_dice[endpoint:endpoint+chsize]]
                endpoint += chsize
                if not cand_selected.empty:
                    cand_selected.to_csv(f'{tmpdir}/{i}.tsv',sep='\t')
                    sel_list.append(f'{tmpdir}/{i}.tsv')
            tsv_merge(sel_list,f'{path.rsplit(".",1)[0]}_downsized.tsv')
        return f'{path.rsplit(".",1)[0]}_downsized.tsv'

    def CandsKernelGenerator(self):
        cands_fr1 = pd.read_table(self.mp_rct1,index_col=0,header=0,chunksize=CHUNK)
        cands_fr2 = pd.read_table(self.mp_rct2,index_col=0,header=0,chunksize=CHUNK)
        self.lgr(f'Load candidates {self.mp_rct1}, {self.mp_rct2}')

        if 'downsize_sc' in self.args:
            self.lgr(f'Downsize option was selected. Size of each reactant: {self.args["downsize_sc"]}')
            mp_rct1_downsized = self.randomselector(self.mp_rct1,'downsize_sc')
            cands_fr1 = pd.read_table(mp_rct1_downsized,index_col=0,header=0,chunksize=CHUNK)
            mp_rct2_downsized = self.randomselector(self.mp_rct2,'downsize_sc')
            cands_fr2 = pd.read_table(mp_rct2_downsized,index_col=0,header=0,chunksize=CHUNK)

        pdict_cand_fr1 = {
            'metadata' : self.mps_rct1,
            'kernel'   : self.mpk_rct1
            }
        pdict_cand_fr2 = {
            'metadata' : self.mps_rct2,
            'kernel'   : self.mpk_rct2
            }
        
        r1_txt = 'Kernel calculation of reactant 1'
        with timer(process_name=r1_txt) as t:
            self.lgr(f'{r1_txt} start.')
            sh_rct_1, sh_dup_rct_1 = self._mod_cands_knl_gen(cands_fr1,self.sv_[:,:8192],pdict_cand_fr1,self.isin_rct1_pool,-1)
            self.lgr(f'---Reactant 1---')
            self.lgr(f'Number of reactants: {sh_rct_1}')
            self.lgr(f'Number of duplicated reactants: {sh_dup_rct_1}')
            tim = t.get_runtime()
            self.lgr(f'{r1_txt} end. Took {tim} sec.')
            self.res_dict['rct1_kernel_calctime'] = tim

        r2_txt = 'Kernel calculation of reactant 2'
        with timer(process_name=r2_txt) as t:
            self.lgr(f'{r2_txt} start.')
            sh_rct_2, sh_dup_rct_2 = self._mod_cands_knl_gen(cands_fr2,self.sv_[:,8192:],pdict_cand_fr2,self.isin_rct2_pool,-1)
            self.lgr(f'---Reactant 2---')
            self.lgr(f'Number of reactants: {sh_rct_2}')
            self.lgr(f'Number of duplicated reactants: {sh_dup_rct_2}') 
            tim = t.get_runtime()
            self.lgr(f'{r2_txt} end. Took {tim} sec.')
            self.res_dict['rct2_kernel_calctime'] = tim
        self.res_dict['possible_combinations'] = sh_rct_1 * sh_rct_2

    def ConstantsExtractor(self):
        # Template that define reaction
        self.template = self.test_rct[self.test_rct['Rep_reaction']==self.reaction]['template'].iloc[0]
        self.rc = reactor(self.template,cumurate=False)
        # Define reaction center
        sub = self.template.split('>>')[1]
        self.center = ReactionCenter(sub,cumurate=False)
        # Reactants pool of train
        rct1_pool = set(self.train_rct['Precursors1'])
        rct2_pool = set(self.train_rct['Precursors2'])
        self.isin_rct1_pool = lambda x: x in rct1_pool
        self.isin_rct2_pool = lambda x: x in rct2_pool
        # Heavy atoms range
        self.rct1_hatoms_range = (
            np.min(self.train_rct['Precursors1_num_of_hatoms']),
            np.max(self.train_rct['Precursors1_num_of_hatoms'])
            )
        self.rct2_hatoms_range = (
            np.min(self.train_rct['Precursors2_num_of_hatoms']),
            np.max(self.train_rct['Precursors2_num_of_hatoms'])
            )

    def LoadKernelsAndComparingDatas(self,file_uni_name,reaction):
        self.PredPathExtractor(file_uni_name,reaction)
        # load model
        self.svr        = self.rxnmlmod.ml_rct_[self.reaction].cv_models_['svr_tanimoto_split'].best_estimator_
        self.sv_        = self.rxnmlmod.ml_rct_[self.reaction].support_vectors_['svr_tanimoto_split']
        self.coef_      = self.rxnmlmod.ml_rct_[self.reaction].cv_models_['svr_tanimoto_split'].best_estimator_.dual_coef_
        self.intercept_ = self.rxnmlmod.ml_rct_[self.reaction].cv_models_['svr_tanimoto_split'].best_estimator_.intercept_[0]
        self.train_rct  = pd.read_table(self.path_rct_train, header=0, index_col=0)
        self.test_rct   = pd.read_table(self.path_rct_test,  header=0, index_col=0)
        self.train_prd  = pd.read_table(self.path_prd_train, header=0, index_col=0)

    def LoadKernelVals(self):
        # load kernels
        kernel_fr1 = np.load(self.mpk_rct1)
        kernel_fr2 = np.load(self.mpk_rct2)
        return kernel_fr1, kernel_fr2
        
    def LoadMod(self,file_uni_name):
        self.ModPathExtractor(file_uni_name)
        self.rxnmlmod = pickle.load(open(self.model_path, 'rb'))

    def LoadScoredMetadata(self):
        # load metadata
        meta_rct1 = pd.read_table(self.mps_rct1,header=0,index_col=0)
        meta_rct2 = pd.read_table(self.mps_rct2,header=0,index_col=0)
        return meta_rct1, meta_rct2

    def ModPathExtractor(self,file_uni_name):
        if 'augmentation' in self.args and self.args['augmentation']:
            self.dir_pred = f'{self.args["out_dir"]}/prediction_level{self.args["split_level"]}_augmented/{file_uni_name}'
        else:
            self.dir_pred = f'{self.args["out_dir"]}/prediction_level{self.args["split_level"]}/{file_uni_name}'
        self.model_path    = f'{self.dir_pred}/mod.pickle'
        self.path_rct_train = f'{self.dir_pred}/prediction_results_rct_train.tsv'
        self.path_rct_test  = f'{self.dir_pred}/prediction_results_rct_test.tsv'
        self.path_prd_train = f'{self.dir_pred}/prediction_results_prd_train.tsv'

    def PredictFromProducts(self,njobs=1,retrieve_size=None):
        self.lgr('Prediction from products using SVR-baseline...')
        self.lgr('Note! This process overwrite the result file.')
        if retrieve_size is not None:
            self.lgr(f'Retrieve size is set to {retrieve_size}.')
            df_to_retrieve = None
        sc_preds_chunked  = pd.read_table(self.passed_,header=0,index_col=0,chunksize=CHUNK)
        sc_preds_paths = []
        self.res_dict['svr_tanimoto_predicted_min'] = np.inf
        self.res_dict['svr_tanimoto_predicted_max'] = -np.inf
        with timer(process_name='Prediction from products') as t:
            with tempfile.TemporaryDirectory() as tmpdir:
                for i,sc_preds in enumerate(sc_preds_chunked):
                    ## Prediction
                    sc_preds_bit = self.rxnmlmod._var_gen(sc_preds,'Product_norxncenter',False,False)
                    ret = Parallel(n_jobs=njobs,backend='threading')(
                        [delayed(self.rxnmlmod.ml_prd_[self.reaction].cv_models_['svr_tanimoto'].best_estimator_.predict)(csr_matrix(arr)) 
                        for arr in np.array_split(sc_preds_bit, os.cpu_count())]
                        )
                    ret_flatten = np.concatenate(ret)
                    if np.min(ret_flatten) < self.res_dict['svr_tanimoto_predicted_min']:
                        self.res_dict['svr_tanimoto_predicted_min'] = np.min(ret_flatten)
                    if np.max(ret_flatten) > self.res_dict['svr_tanimoto_predicted_max']:
                        self.res_dict['svr_tanimoto_predicted_max'] = np.max(ret_flatten)
                    sc_preds['svr_tanimoto_predict'] = ret_flatten
                    if retrieve_size is not None:
                        if df_to_retrieve is None:
                            df_to_retrieve = sc_preds.nlargest(self.args['n_samples'],'svr_tanimoto_predict')
                        else:
                            df_to_retrieve = pd.concat([df_to_retrieve,sc_preds]).nlargest(self.args['n_samples'],'svr_tanimoto_predict')

                    sc_preds.to_csv(f'{tmpdir}/{i}.tsv',sep='\t')
                    sc_preds_paths.append(f'{tmpdir}/{i}.tsv')
                tsv_merge(sc_preds_paths,self.passed_)
                if retrieve_size is not None:
                    df_to_retrieve.to_csv(self.passed_.replace('.tsv','_retrieved.tsv'),sep='\t')
                    self.res_dict['svr_tanimoto_predicted_min'] = np.min(df_to_retrieve['svr_tanimoto_predict'])
                self.res_dict['prediction_time_from_products'] = t.get_runtime()        
        self.lgr('Process completed.')

    def PredPathExtractor(self, file_uni_name, reaction):
        if 'augmentation' in self.args and self.args['augmentation']:
            odir_pref = f'{self.args["out_dir"]}/reactant_combination_level{self.args["split_level"]}_augmented'
        else:
            odir_pref = f'{self.args["out_dir"]}/reactant_combination_level{self.args["split_level"]}'

        if 'downsize_sc' in self.args and self.args['downsize_sc']:
            out_dir_sc_base  = f'{odir_pref}_{self.args["n_samples"]}_rc{self.args["downsize_sc"]}/{file_uni_name}'
        else:
            out_dir_sc_base  = f'{odir_pref}_{self.args["n_samples"]}/{file_uni_name}'
        self.dir_cand = f'{out_dir_sc_base}/{reaction}'
        self.mp_rct1  = f'{os.path.dirname(self.args["cand_path"])}/{file_uni_name}_{reaction}_rct1_candidates_whole.tsv'
        self.mps_rct1 = f'{self.dir_cand}/{file_uni_name}_{reaction}_rct1_candidates_selected_whole.tsv'
        self.mpk_rct1 = f'{self.dir_cand}/{file_uni_name}_{reaction}_rct1_candidates_selected_kernel_whole.npy'
        self.mp_rct2  = f'{os.path.dirname(self.args["cand_path"])}/{file_uni_name}_{reaction}_rct2_candidates_whole.tsv'
        self.mps_rct2 = f'{self.dir_cand}/{file_uni_name}_{reaction}_rct2_candidates_selected_whole.tsv'
        self.mpk_rct2 = f'{self.dir_cand}/{file_uni_name}_{reaction}_rct2_candidates_selected_kernel_whole.npy'
        self.passed_  = f'{self.dir_cand}/{file_uni_name}_{reaction}_rct_candidates_pairs_whole_sparse_split_highscored.tsv'
        MakeDirIfNotExisting(self.dir_cand)

    def ResultsExtractor(self,njobs=1):
        """Extract results of screening
            <Input>
             njobs: int
                Paralyzation
            <Return>
             None, results are saved in specific named files
        """
        self.lgr('Extraction of results start')
        strt_t = time()
        _jobs  = njobs if njobs > 0 else os.cpu_count() - (njobs + 1)
        meta_rct1, meta_rct2 = self.LoadScoredMetadata()

        rct1s = {}
        rct2s = {}

        sec_thres,combs,scores = extract_n_samples_by_threshold(f'{self.dir_cand}/ok_combination.tsv',retrieve_size=self.args['n_samples']*100)
        self.res_dict['second_thres'] = sec_thres
        self.res_dict['combinations_passed_second_threshold'] = combs.shape[0]
        order = np.argsort(scores)[::-1][:self.args['n_samples']*100]
        combs_sorted  = combs[order].astype(int)
        scores_sorted = scores[order]
        for (fr1, fr2) in combs_sorted:
            if fr1 not in rct1s:
                rct1s[fr1] = meta_rct1.iloc[fr1]
            if fr2 not in rct2s:
                rct2s[fr2] = meta_rct2.iloc[fr2]

        knl_fr1s, knl_fr2s = self.LoadKernelVals()
        
        def CombinationExtractor(combs,scores,rct1s,rct2s,rc,knl_fr1s,knl_fr2s,coef_,intercept_,outfname):
            combs_columns = ['Precursors', 'Product', 'Product_norxncenter', 'kernel_max', 'pred_obj']
            combs_split   = np.array_split(combs,_jobs)
            def _mod_combs_ext(combs,rct1s,rct2s,rc,knl_fr1s,knl_fr2s,coef_,intercept_,worker_id):
                combs_index   = []
                combs_data    = []
                for i,((fr1, fr2), _) in enumerate(zip(combs,scores)):
                    rc_copy = deepcopy(rc)
                    if worker_id == 0: # monitoring progress
                        print(f'Processing batch {i}/{len(combs)}', end='\r')
                    fr1_md = rct1s[fr1]
                    fr2_md = rct2s[fr2]
                    knl_fr1 = knl_fr1s[fr1]
                    knl_fr2 = knl_fr2s[fr2]
                    prd = rc_copy.reactor(
                        Chem.MolFromSmiles(fr1_md['mapped_smiles']),
                        Chem.MolFromSmiles(fr2_md['mapped_smiles'])
                        )
                    combs_index.append(f"{fr1_md[self.args['cand_index_col']]},{fr2_md[self.args['cand_index_col']]}")
                    tmp_list = [
                        f"{fr1_md['mapped_smiles']}.{fr2_md['mapped_smiles']}",
                        rc_copy.prd_iso,
                        prd,
                        np.max(knl_fr1 * knl_fr2),
                        np.sum(coef_ * knl_fr1 * knl_fr2) + intercept_,
                    ]
                    combs_data.append(tmp_list)
                return combs_index, combs_data
            ret = Parallel(n_jobs=_jobs,backend='threading')(
                [delayed(_mod_combs_ext)(comb,rct1s,rct2s,rc,knl_fr1s,knl_fr2s,coef_,intercept_,idx)
                for idx, comb in enumerate(combs_split)]
            )
            ret_T = list(zip(*ret))
            combs_index = [x for row in ret_T[0] for x in row]
            combs_data  = [x for row in ret_T[1] for x in row]
            df_valid = pd.DataFrame(combs_data,index=combs_index,columns=combs_columns)
            df_valid.to_csv(outfname,sep='\t')
            return df_valid.shape[0]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            combs_split    = ArraySplitByN(combs_sorted,int(self.args['n_samples']/2))
            scores_split   = ArraySplitByN(scores_sorted,int(self.args['n_samples']/2))
            outfnames = [f'{tmpdir}/{i}.tsv' for i in range(int(np.ceil(len(combs_sorted)/int(self.args['n_samples']/2))))]
            ret = [CombinationExtractor(comb,score,rct1s,rct2s,self.rc,knl_fr1s,knl_fr2s,self.coef_,self.intercept_,outfname)
                for idx, (comb, score, outfname) in tqdm(enumerate(zip(combs_split,scores_split,outfnames)))]
            tsv_merge(outfnames,self.passed_)
        self.lgr(f'Num of extracted combinations: {sum(ret)}')
        self.res_dict['combinations_cpd_filter'] = sum(ret)
        tim = time()-strt_t
        self.lgr(f'Extraction end. Took {tim} sec.')
        self.res_dict['extraction_time'] = tim
        return
    
    def RouteExaminator(self,retrieved=False):
        self.lgr('Checking the suggested route...')
        self.lgr('This process may takes long.')
        passed_ = self.passed_.replace('.tsv','_retrieved.tsv') if retrieved else self.passed_
        p, shape = ValidityAndSuggestedRouteExaminator(passed_,f'{passed_.rsplit(".",1)[0]}_route.tsv',
                                 input_smi_col='Product_norxncenter',prec_smi_col='Precursors')
        self.lgr('Check completed.')
        self.lgr(f'{shape} compounds are remained.')
        self.res_dict['combinations_retro_filter'] = shape

    def SubmatchAndMapping(self, cands, arg_rct1, arg_rct2):
        """Substructure match and reaction center map
            <Input>
             cands: pd.DataFrame, must be chunked
                Dataset for retrieving reactants
             arg_rct1 : tuple
                conditions for substructure match
             arg_rct2 : tuple
                conditions for substructure match
            <Return>
             None, results are saved in specific named files
        """
        plist_fr1 = []
        uni_fr1   = set()
        plist_fr2 = []
        uni_fr2   = set()
        sub_txt   = 'Substructure matching and mapping'
        with timer(process_name=sub_txt) as t:
            self.lgr(f'{sub_txt} start.')
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, cand in enumerate(cands):
                    with timer(process_name=f'Process {i}'):
                        p_fr1 = f'{tmpdir}/fr1_{i}.tsv'
                        self._mod_submatch_and_mapping(cand, self.rct1_hatoms_range, arg_rct1, p_fr1, uni_fr1)
                        plist_fr1.append(p_fr1)

                        p_fr2 = f'{tmpdir}/fr2_{i}.tsv'
                        self._mod_submatch_and_mapping(cand, self.rct2_hatoms_range, arg_rct2, p_fr2, uni_fr2)
                        plist_fr2.append(p_fr2)

                tsv_merge(plist_fr1,merge_file=self.mp_rct1)
                tsv_merge(plist_fr2,merge_file=self.mp_rct2)
            self.lgr(f'{sub_txt} end. Took {t.get_runtime()} sec.')

    def SVRpredCombinations(self, ext_ratio=1e-5, njobs=-1):
        """Screening by ProductTanimotoKernel
            <Input>
             ext_ratio: float
                Determine the number of combinations to be extracted
             njobs: int
                Paralyzation
            <Return>
             None, results are saved in specific named files
        """
        # settings
        njobs = cpu_count() -1 if njobs < 0 else njobs
        kmat1, kmat2 = self.LoadKernelVals() # loading all data
        nfr1  = len(kmat1)
        nfr2  = len(kmat2)
        self.lgr(f'Screening {nfr1*nfr2} combinations.')

        self.res_dict['combinations_to_be_retrieved'] = self.args['n_samples']
        kmat1_sample, kmat2_sample = NeatCombinationExtractor(kmat1, kmat2, self.args['n_samples'])
        ext_indice = np.max([int((kmat1_sample.shape[0] * kmat2_sample.shape[0]) * ext_ratio),1])
        kmat2s_sample = np.array_split(kmat2_sample, njobs)
        mcumurateshape_array = EachArrayLengthCalclator(kmat2s_sample)
        ret = Parallel(n_jobs=njobs, backend='threading')(
            [delayed(_mod_svrpred_extract)(kmat1_sample, submat2, self.coef_, self.intercept_, idx) 
            for idx,submat2 in enumerate(kmat2s_sample)])
        ret_with_shape = ShapeConcatenator(ret,mcumurateshape_array)
        combs_sample_array = np.concatenate(ret_with_shape,axis=0)
        thres_score = sorted(combs_sample_array[:,-1].ravel())[-ext_indice] \
            if len(combs_sample_array) > ext_indice else max(combs_sample_array[:,-1].ravel())
        self.lgr(f'Threshould score: {thres_score}')
        self.res_dict['initial_threshold'] = thres_score

        strt_time = time()

        if nfr1 * nfr2 > self.args['n_samples']:
            kmat2s = np.array_split(kmat2, njobs)
            mcumurateshape_array = EachArrayLengthCalclator(kmat2s)

            with tempfile.TemporaryDirectory() as tmpdir:
                ret = Parallel(n_jobs=njobs, backend='threading')(
                    [delayed(_mod_svrpred_extract)(kmat1, submat2, self.coef_, self.intercept_, idx, thres_score, f'{tmpdir}/{idx}.tsv', self.lgr) 
                    for idx,submat2 in enumerate(kmat2s)])
                ret_with_shape = ShapeConcatenator(ret,mcumurateshape_array,True)
            
            self.lgr(f'Unpacking solutions...')
            combs_array = np.concatenate(ret_with_shape,axis=0)
        else:
            self.res_dict['initial_threshold'] = None
            combs_array = combs_sample_array

        ntotal = combs_array.shape[0]
        self.lgr(f'Total number of combinations: {ntotal}')
        self.res_dict['combinations_passed_initial_threshold'] = ntotal
        np.savetxt(f'{self.dir_cand}/ok_combination.tsv', combs_array, delimiter='\t')
        tim = time() - strt_time
        self.lgr(f'Execution time: {tim} sec.')
        self.res_dict['evaluation_time'] = tim
    
    def SVRpredAnalysis(self, ext_ratios=[5e-5,1e-5,5e-6,1e-6], njobs=-1):
        """Screening by ProductTanimotoKernel
            <Input>
             njobs: int
                Paralyzation
            <Return>
             None, results are saved in specific named files
        """
        ext_ratios_dict = {
            ext: {'indice': None,'threshold': None, 'shape': 0}
            for ext in ext_ratios
        }
        # settings
        njobs = cpu_count() -1 if njobs < 0 else njobs
        kmat1, kmat2 = self.LoadKernelVals() # loading all data
        nfr1  = len(kmat1)
        nfr2  = len(kmat2)
        self.lgr(f'Screening {nfr1*nfr2} combinations.')
        
        kmat1_sample, kmat2_sample = NeatCombinationExtractor(kmat1, kmat2, self.args['n_samples'])
        kmat2s_sample = np.array_split(kmat2_sample, njobs)
        mcumurateshape_array = EachArrayLengthCalclator(kmat2s_sample)
        ret = Parallel(n_jobs=njobs, backend='threading')(
            [delayed(_mod_svrpred_extract)(kmat1_sample, submat2, self.coef_, self.intercept_, idx) 
            for idx,submat2 in enumerate(kmat2s_sample)])
        ret_with_shape = ShapeConcatenator(ret,mcumurateshape_array)
        combs_sample_array = np.concatenate(ret_with_shape,axis=0)
        for ratio in ext_ratios_dict.keys():
            if ratio is not None:
                ext_ratios_dict[ratio]['indice']    = int((kmat1_sample.shape[0] * kmat2_sample.shape[0]) * ratio)
                ext_ratios_dict[ratio]['threshold'] = sorted(combs_sample_array[:,-1].ravel())[-ext_ratios_dict[ratio]['indice']] \
                    if len(combs_sample_array) > ext_ratios_dict[ratio]['indice'] else min(combs_sample_array[:,-1].ravel())
                self.lgr(f'Threshould score (ratio={ratio}): {ext_ratios_dict[ratio]["threshold"]}')

        strt_time = time()

        if nfr1 * nfr2 > self.args['n_samples']:
            kmat2s = np.array_split(kmat2, njobs)
            mcumurateshape_array = EachArrayLengthCalclator(kmat2s)

            ret = Parallel(n_jobs=njobs, backend='threading')(
                [delayed(_mod_svrpred_analysis)(kmat1, submat2, self.coef_, self.intercept_, idx, ext_ratios_dict) 
                for idx,submat2 in enumerate(kmat2s)])
            for r in ret:
                for ratio, shape in r.items():
                    ext_ratios_dict[ratio]['shape'] = ext_ratios_dict[ratio]['shape'] + shape
            self.lgr(f'Unpacking solutions...')
        else:
            for ratio in ext_ratios_dict.keys():
                if ratio is not None:
                    ext_ratios_dict[ratio]['shape'] = len(np.where(combs_sample_array[:,-1].ravel() >= ext_ratios_dict[ratio]['threshold'])[0])
                else:
                    ext_ratios_dict[ratio]['shape'] = len(combs_sample_array[:,-1].ravel())
        for ratio, d in ext_ratios_dict.items():
            self.lgr(f'Total number of combinations (ratio={ratio}): {d["shape"]}')
        self.lgr(f'Execution time: {time() - strt_time} sec.')

    def close_log(self):
        if self.logger is not None:
            close_logger(self.logger)

