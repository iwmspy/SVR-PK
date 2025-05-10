''' modeling'''

from copy import deepcopy
from dataclasses import dataclass
import os
from typing import Optional, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack
from sklearn import svm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from models._kernel_and_mod import *
from utils.utility import timer
from utils.chemutils import MorganbitCalcAsVectors,MurckoScaffoldSmilesListFromSmilesList

unique_scaf_length = lambda l,s: str([np.unique(row).size for row in MurckoScaffoldSmilesListFromSmilesList(l,split_components=s).T])

def to_Series(df:pd.DataFrame, index_col:str, val_col:str):
    return pd.Series(data=df[val_col].to_list(),
                     index=df[index_col],
                     name=val_col)

def _dfpad(df_to_pad_1: pd.DataFrame, df_to_pad_2: pd.DataFrame):
    cnt = 0
    while df_to_pad_1.shape[0] < df_to_pad_2.shape[0]:
        df_to_pad_1.loc[f'_{cnt}'] = '-'
        df_to_pad_1.loc[f'_{cnt}','num_sample'] = df_to_pad_1.iloc[0]['num_sample']
        cnt += 1

def DataFramePadding(df_to_pad_1: pd.DataFrame, df_to_pad_2: pd.DataFrame):
    _dfpad(df_to_pad_1, df_to_pad_2)
    _dfpad(df_to_pad_2, df_to_pad_1)

def AnalyzeSomeStatisticalValues(l: list):
    if not len(l): return dict(
            mean=-1,
            std =-1,
            range=(-1,-1)
        )
    return dict(
        mean=round(np.mean(l),3),
        std =round(np.std(l),3),
        range=(np.min(l),np.max(l))
    )

@dataclass
class ModelParams():
	name: str
	model: Any
	cv_scoring: str				 = 'r2'
	cv_params: Optional[dict]	 = None
	fixed_params: Optional[dict] = None


model_param_template = {
	'svr_linear'           : ModelParams(name='svr_linear', 	
        model=svm.SVR, 
        cv_params=dict(C=np.logspace(-4, 4, 9)),
        fixed_params=dict(kernel='linear', max_iter=10000)
        ),
	'lasso_cv'             : ModelParams(name='lasso_cv', 	
        model=LassoCV, 
        fixed_params=dict(random_state=0, max_iter=10000)
        ),
	'svr_tanimoto'         : ModelParams(name='svr_tanimoto', 	
        model=svm.SVR, 
        cv_params=dict(
            C=np.logspace(-3, 3, num=7, base=2),
            epsilon=np.logspace(-10, 0, num=11, base=2)),
        fixed_params=dict(kernel=funcTanimotoSklearn)
        ),
    'svr_tanimoto_product'   : ModelParams(name='svr_tanimoto_product', 	
        model=svm.SVR, 
        cv_params=dict(
            C=np.logspace(-3, 3, num=7, base=2),
            epsilon=np.logspace(-10, 0, num=11, base=2)),
        fixed_params=dict(kernel=ProductTanimotoKernel)
        ),
    'svr_tanimoto_split'     : ModelParams(name='svr_tanimoto_product', 	# alias of 'svr_tanimoto_product'
        model=svm.SVR, 
        cv_params=dict(
            C=np.logspace(-3, 3, num=7, base=2),
            epsilon=np.logspace(-10, 0, num=11, base=2)),
        fixed_params=dict(kernel=ProductTanimotoKernel)
        ),
    'svr_tanimoto_average' : ModelParams(name='svr_tanimoto_average', 	
        model=svm.SVR, 
        cv_params=dict(
            C=np.logspace(-3, 3, num=7, base=2),
            epsilon=np.logspace(-10, 0, num=11, base=2)),
        fixed_params=dict(kernel=AverageTanimotoKernel)
        ),
	'rf'                   : ModelParams(name='random_forest', 	
        model=RandomForestRegressor, 	
        cv_params=dict(n_estimators=[50, 100, 300, 500],  max_features=[None, 'sqrt', 'log2'], 
                       max_depth = [10, 30, 50, None]),
        fixed_params=dict(random_state=0)
        ),
    'adaboost'             : ModelParams(name='adaboost', 	
        model=AdaBoostRegressor, 	
        cv_params=dict(n_estimators=[10, 100, 500], learning_rate=[0.2, 0.6, 1.0]),
        fixed_params=dict(random_state=0)
        ),
	}


def _get_params(plist:list='all'):
    if plist=='all':
        return deepcopy(model_param_template)
    else:
        assert isinstance(plist,list)
        ps = dict()
        for p in plist:
            ps[p] = model_param_template[p]
        return ps


class ModelingModule:
    def __init__(self, mls:list='all', random_state:int=0, njobs:int=-1, cv:int=5) -> None:
        self.state = random_state
        self.core = njobs if njobs > 0 else os.cpu_count() - (njobs + 1)
        self.cv   = cv
        self.mls  = _get_params(mls)
    
    def fit(self, minf:ModelParams, X, y):
        if minf.name != 'lasso_cv':
            grid = GridSearchCV(estimator = minf.model(**minf.fixed_params),
                                param_grid = minf.cv_params,
                                scoring = minf.cv_scoring,
                                cv = KFold(n_splits=self.cv,shuffle=True,
                                        random_state=self.state),
                                n_jobs = self.core)
        else:
            grid = minf.model(**minf.fixed_params, cv=self.cv, n_jobs = self.core)
        return grid.fit(X, y)
    
    def run_cv(self, X, y):
        self.cv_models_ = dict()
        self.support_vectors_ = dict()
        for mname, minf in self.mls.items():
            with timer(f'Building of model {mname}'):
                self.cv_models_[mname] = self.fit(minf, X, y)
            if hasattr(self.cv_models_[mname],'best_estimator_'):
                if hasattr(self.cv_models_[mname].best_estimator_,'support_'):
                    self.support_vectors_[mname] = X[self.cv_models_[mname].best_estimator_.support_]
        return deepcopy(self)
    
    def scoring(self, X=None, y:pd.Series=None):
        scores = dict()
        preds  = list()
        for mname, model in self.cv_models_.items():
            bpar = model.best_params_ if mname!='lasso_cv' else model.get_params()
            if X is not None:
                pred = pd.Series(model.predict(X), index=y.index, name=f'{mname}_prd_pred')
                preds.append(pred)
                df   = pd.concat([y,pred],axis=1)
                scores[mname] = pd.Series(dict(
                    num_sample=X.shape[0],
                    best_params=str(bpar),
                    r2=r2_score(df.iloc[:,0],df.iloc[:,1]),
                    rmse=np.sqrt(mean_squared_error(df.iloc[:,0],df.iloc[:,1])),
                    mae=mean_absolute_error(df.iloc[:,0],df.iloc[:,1])
                    ))
            else:
                scores[mname] = pd.Series(dict(
                    num_sample=0,
                    best_params=str(bpar),
                    r2=None,
                    rmse=None,
                    mae=None
                    ))
        scores_df = pd.DataFrame(scores).T
        scores_df.index.name = 'model'
        if X is not None:
            return scores_df, pd.concat(preds,axis=1)
        else:
            return scores_df
    

class ReactantModeling(ModelingModule):
    def __init__(self, mls:list='all', random_state:int=0, njobs:int=-1, cv:int=5) -> None:
        super().__init__(mls, random_state, njobs, cv)
    
    def scoring(self, X=None, y:pd.Series=None, grouping:bool=True):
        scores = dict()
        preds  = list()
        for mname, model in self.cv_models_.items():
            bpar = model.best_params_ if mname!='lasso_cv' else model.get_params()
            if X is not None:
                with timer(f'Scoring of model {mname}'):
                    pred = pd.Series(model.predict(X), index=y.index, name=f'{mname}_rct_pred')
                preds.append(pred)
                df   = pd.concat([y,pred],axis=1)
                df_m = df.groupby(level=0).mean() if grouping else df.copy()
                scores[mname] = pd.Series(dict(
                    num_sample=X.shape[0],
                    best_params=str(bpar),
                    r2=r2_score(df_m.iloc[:,0],df_m.iloc[:,1]),
                    rmse=np.sqrt(mean_squared_error(df_m.iloc[:,0],df_m.iloc[:,1])),
                    mae=mean_absolute_error(df_m.iloc[:,0],df_m.iloc[:,1])
                    ))
            else:
                scores[mname] = pd.Series(dict(
                    num_sample=0,
                    best_params=str(bpar),
                    r2=None,
                    rmse=None,
                    mae=None
                    ))
        scores_df = pd.DataFrame(scores).T
        scores_df.index.name = 'model'
        if X is not None:
            return scores_df, pd.concat(preds,axis=1)
        else:
            return scores_df


class ProductModeling(ModelingModule):
    def __init__(self, mls:list='all', random_state:int=0, njobs:int=-1, cv:int=5) -> None:
        super().__init__(mls, random_state, njobs, cv)


class ReactionGroupWrapperModeling:
    def __init__(self, mls_product:list='all', mls_reactant:list='all',
                 random_state:int=0, njobs:int=-1, cv:int=5,
                 outdir='results') -> None:
        self.state = random_state
        self.core = njobs if njobs > 0 else os.cpu_count() - (njobs + 1)
        self.cv   = cv
        self.mls_prd = mls_product
        self.mls_rct = mls_reactant
        self.rm   = ReactantModeling(self.mls_rct, self.state, self.core, self.cv)
        self.pm   = ProductModeling(self.mls_prd, self.state, self.core, self.cv)
        self.odir = outdir
        if not(os.path.exists(self.odir)): os.makedirs(self.odir)
    
    def _var_gen(self,df,smiles_col,split_components,sparse=True):
        var = np.array(
            MorganbitCalcAsVectors(
                df[smiles_col],
                n_jobs=self.core,
                split_components=split_components
                )).astype(bool)
        if sparse: return csr_matrix(var)
        return var        
    
    def _variable_generator(self,df_prd,df_rct,product_smiles_col,reactant_smiles_col,split_components):
        prd_x = self._var_gen(df_prd, product_smiles_col, split_components=False)
        rct_x = self._var_gen(df_rct, reactant_smiles_col, split_components=split_components)
        return prd_x, rct_x
    
    def _save_results(self, score_prd, df_prd_pred, score_rct=None, df_rct_pred=None, which='train'):
        score_prd.to_csv(os.path.join(self.odir,f'prediction_score_prd_{which}.tsv'),sep='\t')
        df_prd_pred.to_csv(os.path.join(self.odir,f'prediction_results_prd_{which}.tsv'),sep='\t')
        if score_rct is not None:
            score_rct.to_csv(os.path.join(self.odir,f'prediction_score_rct_{which}.tsv'),sep='\t')
        if df_rct_pred is not None:
            df_rct_pred.to_csv(os.path.join(self.odir,f'prediction_results_rct_{which}.tsv'),sep='\t')
        
    def run_cv(self,
               df: pd.DataFrame, 
               obj_col: str, 
               product_smiles_col: str, 
               reactant_smiles_col: str, 
               group_index_col: str, 
               group_subgroup_col: str=None,
               df_whole: pd.DataFrame=None,
               split_components:bool=True):
        if group_subgroup_col==None:
            df_prd  = df.drop_duplicates(group_index_col)
            df_rct  = df.copy()
            sr_prd_y = df_prd[obj_col].copy()
            sr_rct_y = to_Series(df_rct,group_index_col,obj_col)
            prd_x = self._var_gen(df_prd,product_smiles_col,False)
            rct_x = self._var_gen(df_rct,reactant_smiles_col,split_components)

            self.ml_prd_ = {'reaction': self.pm.run_cv(prd_x,sr_prd_y)}
            self.ml_rct_ = {'reaction': self.rm.run_cv(rct_x,sr_rct_y)}
        
        else:
            self.template_ = dict()
            self.ml_prd_ = dict()
            self.ml_rct_ = dict()

            if df_whole is not None: 
                self.ml_prd_whole_ = dict()

            for subgroup, subgroup_df in df.groupby(group_subgroup_col):
                print(f'Subset: {subgroup}')
                df_sub_prd   = subgroup_df.drop_duplicates(group_index_col)
                df_sub_rct   = subgroup_df.copy()
                sr_sub_prd_y = df_sub_prd[obj_col].copy()
                sr_sub_rct_y = to_Series(df_sub_rct,group_index_col,obj_col)
                prd_sub_x    = self._var_gen(
                    df_sub_prd,product_smiles_col,False)
                rct_sub_x = self._var_gen(
                    df_sub_rct,reactant_smiles_col,split_components)

                self.template_[subgroup] = df_sub_prd['template'].iloc[0]
                self.ml_prd_[subgroup] = self.pm.run_cv(prd_sub_x,sr_sub_prd_y)
                self.ml_rct_[subgroup] = self.rm.run_cv(rct_sub_x,sr_sub_rct_y)
                if df_whole is not None:
                    df_sub_prd_whole   = df_whole[df_whole[group_subgroup_col]==subgroup].drop_duplicates(group_index_col)
                    sr_sub_prd_whole_y = df_sub_prd_whole[obj_col].copy()
                    prd_whole_sub_x    = self._var_gen(df_sub_prd_whole,product_smiles_col,False)
                    self.ml_prd_whole_[subgroup] = self.pm.run_cv(prd_whole_sub_x, sr_sub_prd_whole_y)

        return self.scoring(df,obj_col,product_smiles_col,reactant_smiles_col,
            group_index_col,group_subgroup_col,df_whole,split_components,_train=True)
    
    def scoring(self,
               df: pd.DataFrame, 
               obj_col: str, 
               product_smiles_col: str, 
               reactant_smiles_col: str, 
               group_index_col: str, 
               group_subgroup_col: str=None,
               df_whole: pd.DataFrame=None,
               split_components:bool=True,
               _train: bool=False):
        _which = 'train' if _train else 'test'
        if group_subgroup_col==None:
            df_prd  = df.drop_duplicates(group_index_col)
            df_rct  = df.copy()
            sr_prd_y = df_prd[obj_col].copy()
            sr_rct_y = to_Series(df_rct,group_index_col,obj_col)
            prd_x, rct_x = self._variable_generator(
                df_prd,df_rct,product_smiles_col,reactant_smiles_col,split_components)

            score_prd, pred_prd = self.ml_prd_['reaction'].scoring(prd_x,sr_prd_y)
            score_rct, pred_rct = self.ml_rct_['reaction'].scoring(rct_x,sr_rct_y)
            pred_rct.set_index(df_rct.index,drop=True,inplace=True)
            df_prd_pred = pd.concat([df_prd,pred_prd],axis=1)
            df_rct_pred = pd.concat([df_rct,pred_rct],axis=1)
            DataFramePadding(score_prd, score_rct)
            self._save_results(score_prd, score_rct, df_prd_pred, df_rct_pred, 'test')
            return score_prd, score_rct
        
        else:
            is_first = True
            for subgroup, subgroup_df in df.groupby(group_subgroup_col):
                print(f'Subset: {subgroup}')
                df_sub_prd   = subgroup_df.drop_duplicates(group_index_col)
                df_sub_rct   = subgroup_df.copy()
                sr_sub_prd_y = df_sub_prd[obj_col].copy()
                sr_sub_rct_y = to_Series(df_sub_rct,group_index_col,obj_col)
                prd_sub_x, rct_sub_x = self._variable_generator(
                    df_sub_prd,df_sub_rct,product_smiles_col,reactant_smiles_col,split_components)

                score_sub_prd, pred_sub_prd = self.ml_prd_[subgroup].scoring(prd_sub_x,sr_sub_prd_y)
                score_sub_rct, pred_sub_rct = self.ml_rct_[subgroup].scoring(rct_sub_x,sr_sub_rct_y)
                DataFramePadding(score_sub_prd, score_sub_rct)
                if _which == 'test':
                    if is_first:
                        self.best_subgroup_estimators_index_ = {ml : (subgroup,score_sub_rct.loc[ml, 'r2']) for ml in self.mls_rct}
                    else:
                        for ml in self.mls_rct:
                            r2 = score_sub_rct.loc[ml, 'r2']
                            if r2 > self.best_subgroup_estimators_index_[ml][1]:
                                self.best_subgroup_estimators_index_[ml] = (subgroup,r2)
                if df_whole is not None:
                    df_sub_prd_whole   = df_whole[df_whole[group_subgroup_col]==subgroup].drop_duplicates(group_index_col)
                    sr_sub_prd_whole_y = df_sub_prd_whole[obj_col].copy()
                    prd_whole_sub_x    = self._var_gen(df_sub_prd_whole,product_smiles_col,False)
                    score_sub_prd_whole, pred_sub_prd_whole = self.ml_prd_whole_[subgroup].scoring(prd_whole_sub_x,sr_sub_prd_whole_y)
                    DataFramePadding(score_sub_prd, score_sub_prd_whole)
                
                # Process results
                # Product
                score_sub_prd['Rep_reaction'] = subgroup
                score_sub_prd['template'] = self.template_[subgroup]
                score_sub_prd['data_statistics'] = str(AnalyzeSomeStatisticalValues(sr_sub_prd_y))
                score_sub_prd['num_unique_scaf'] = unique_scaf_length(df_sub_prd[product_smiles_col].to_list(),False)
                score_sub_prd.set_index(['Rep_reaction','template','num_sample','num_unique_scaf','data_statistics',score_sub_prd.index],inplace=True)
                score_prd = score_sub_prd.copy() \
                    if is_first else pd.concat([score_prd, score_sub_prd])
                df_prd_pred = pd.concat([df_sub_prd,pred_sub_prd],axis=1) \
                    if is_first else pd.concat(
                        [df_prd_pred, pd.concat([df_sub_prd,pred_sub_prd],axis=1)])
                if df_whole is not None:
                    score_sub_prd_whole['Rep_reaction'] = subgroup
                    score_sub_prd_whole['template'] = self.template_[subgroup]
                    score_sub_prd_whole['data_statistics'] = str(AnalyzeSomeStatisticalValues(sr_sub_prd_whole_y))
                    score_sub_prd_whole.set_index(['Rep_reaction','template','num_sample','data_statistics',score_sub_prd_whole.index],inplace=True)
                    score_prd_whole = score_sub_prd_whole.copy() \
                        if is_first else pd.concat([score_prd_whole, score_sub_prd_whole])
                    df_prd_pred_whole = pd.concat([df_sub_prd,pred_sub_prd_whole],axis=1) \
                        if is_first else pd.concat(
                            [df_prd_pred_whole, pd.concat([df_sub_prd,pred_sub_prd_whole],axis=1)])
                # Reactant
                score_sub_rct['Rep_reaction'] = subgroup
                score_sub_rct['template'] = self.template_[subgroup]
                score_sub_rct['data_statistics'] = str(AnalyzeSomeStatisticalValues(sr_sub_rct_y))
                score_sub_rct['num_unique_scaf'] = unique_scaf_length(df_sub_rct[reactant_smiles_col].to_list(),True)
                score_sub_rct.set_index(['Rep_reaction','template','num_sample','num_unique_scaf','data_statistics',score_sub_rct.index],inplace=True)
                pred_sub_rct.set_index(df_sub_rct.index,drop=True,inplace=True)
                score_rct = score_sub_rct.copy() \
                    if is_first else pd.concat([score_rct, score_sub_rct])
                df_rct_pred = pd.concat([df_sub_rct,pred_sub_rct],axis=1) \
                    if is_first else pd.concat(
                        [df_rct_pred, pd.concat([df_sub_rct,pred_sub_rct],axis=1)])
                is_first = False
            not_scored = [x for x in self.ml_prd_.keys() if x not in set(df[group_subgroup_col])]
            for ns in not_scored:
                print(f'Subset: {ns}')
                score_sub_prd = self.ml_prd_[ns].scoring(None,None)                
                score_sub_rct = self.ml_rct_[ns].scoring(None,None)
                DataFramePadding(score_sub_rct, score_sub_prd)
                score_sub_prd['Rep_reaction'] = ns
                score_sub_prd['template'] = self.template_[ns]
                score_sub_prd['num_unique_scaf'] = str([0])
                score_sub_prd['data_statistics'] = str(AnalyzeSomeStatisticalValues([]))
                score_sub_prd.set_index(['Rep_reaction','template','num_sample','num_unique_scaf','data_statistics',score_sub_prd.index],inplace=True)
                score_prd = pd.concat([score_prd, score_sub_prd])
                score_sub_rct['Rep_reaction'] = ns
                score_sub_rct['template'] = self.template_[ns]
                score_sub_rct['num_unique_scaf'] = str([0, 0])
                score_sub_rct['data_statistics'] = str(AnalyzeSomeStatisticalValues([]))
                score_sub_rct.set_index(['Rep_reaction','template','num_sample','num_unique_scaf','data_statistics',score_sub_rct.index],inplace=True)
                score_rct = pd.concat([score_rct, score_sub_rct])
                if df_whole is not None:
                    score_sub_prd_whole = self.ml_prd_whole_[ns].scoring(None, None)
                    DataFramePadding(score_sub_prd, score_sub_prd_whole)
                    score_sub_prd_whole['Rep_reaction'] = ns
                    score_sub_prd_whole['template'] = self.template_[ns]
                    score_sub_prd_whole['data_statistics'] = str(AnalyzeSomeStatisticalValues([]))
                    score_sub_prd_whole.set_index(['Rep_reaction','template','num_sample','data_statistics',score_sub_prd_whole.index],inplace=True)
                    score_prd_whole = pd.concat([score_prd_whole, score_sub_prd_whole])
                    
            self._save_results(score_prd, df_prd_pred, score_rct, df_rct_pred, which=f'{_which}')
            if df_whole is None:
                return score_prd, score_rct
            self._save_results(score_prd_whole, df_prd_pred_whole,which=f'{_which}_whole')
            return score_prd, score_prd_whole, score_rct


class ReactionGroupWrapperModelingWithRadius(ReactionGroupWrapperModeling):
    def __init__(self, mls_product:list='all', mls_reactant:list='all',
                 rad:int=2, nbits:int=8192,
                 random_state:int=0, njobs:int=-1, cv:int=5,
                 outdir='results') -> None:
        super().__init__(mls_product,mls_reactant,random_state,njobs,cv,outdir)
        self.rad  = rad
        self.nbit = nbits
            
    def _var_gen(self,df,smiles_col,split_components,sparse=True):
        var = np.array(
            MorganbitCalcAsVectors(
                df[smiles_col],
                rad=self.rad,
                bits=self.nbit,
                n_jobs=self.core,
                split_components=split_components
                )).astype(bool)
        if sparse: return csr_matrix(var)
        return var        


class ReactionWholeWrapperModeling(ReactionGroupWrapperModeling):
    def __init__(self, 
                 mls_product: list = 'all',
                 mls_reactant: list = 'all', 
                 rad: int = 2, 
                 nbits: int = 8192, 
                 random_state: int = 0, 
                 njobs: int = -1, 
                 cv: int = 5, 
                 outdir='results') -> None:
        super().__init__(mls_product, mls_reactant, rad, nbits, random_state, njobs, cv, outdir)
        
    def run_cv(self,
               df: pd.DataFrame, 
               obj_col: str, 
               product_smiles_col: str, 
               reactant_smiles_col: str, 
               group_index_col: str, 
               group_subgroup_col: str=None,
               df_whole: pd.DataFrame=None,
               split_components:bool=True):
        df_prd  = df.drop_duplicates(group_index_col)
        df_rct  = df.copy()
        sr_prd_y = df_prd[obj_col].copy()
        sr_rct_y = to_Series(df_rct,group_index_col,obj_col)
        self.prd_set  = {'smiles': df_prd[product_smiles_col].to_list()}
        self.prd_set_var = self._var_gen(self.prd_set,'smiles',False)
        self.rct_set  = {'smiles': sorted(list(set([x for row in [rct.split('.') for rct in df_rct[reactant_smiles_col]] for x in row])))}
        rct_set_var_raw  = self._var_gen(self.rct_set,'smiles',False)
        self.rct_set_var = hstack([rct_set_var_raw, rct_set_var_raw])
        prd_x_raw = self._var_gen(df_prd,product_smiles_col,False)
        rct_x_raw = self._var_gen(df_rct,reactant_smiles_col,split_components)
        prd_x = funcTanimotoSklearn(prd_x_raw, self.prd_set_var,n_jobs=self.core)
        rct_x = AverageTanimotoKernel(rct_x_raw, self.rct_set_var,n_jobs=self.core)

        ml_prd_ = self.pm.run_cv(prd_x,sr_prd_y)
        ml_rct_ = self.rm.run_cv(rct_x,sr_rct_y)

        if group_subgroup_col is None:
            self.ml_prd_ = {'reaction': ml_prd_}
            self.ml_rct_ = {'reaction': ml_rct_}
        else:
            self.template_ = {subgroup: subgroup_df['template'].iloc[0] for subgroup, subgroup_df in df_rct.groupby(group_subgroup_col)}
            self.ml_prd_ = {subgroup: deepcopy(ml_prd_) for subgroup in set(df_prd[group_subgroup_col])}
            self.ml_rct_ = {subgroup: deepcopy(ml_rct_) for subgroup in set(df_rct[group_subgroup_col])}

        return self.scoring(df,obj_col,product_smiles_col,reactant_smiles_col,
            group_index_col,group_subgroup_col,df_whole,split_components,_train=True)
    
    def scoring(self,
               df: pd.DataFrame, 
               obj_col: str, 
               product_smiles_col: str, 
               reactant_smiles_col: str, 
               group_index_col: str, 
               group_subgroup_col: str=None,
               df_whole: pd.DataFrame=None,
               split_components:bool=True,
               _train: bool=False):
        _which = 'train' if _train else 'test'
        if group_subgroup_col==None:
            df_prd  = df.drop_duplicates(group_index_col)
            df_rct  = df.copy()
            sr_prd_y = df_prd[obj_col].copy()
            sr_rct_y = to_Series(df_rct,group_index_col,obj_col)
            prd_x_raw, rct_x_raw = self._variable_generator(
                df_prd,df_rct,product_smiles_col,reactant_smiles_col,split_components)
            prd_x = funcTanimotoSklearn(prd_x_raw,self.prd_set_var)
            rct_x = AverageTanimotoKernel(rct_x_raw,self.rct_set_var)

            score_prd, pred_prd = self.ml_prd_['reaction'].scoring(prd_x,sr_prd_y)
            score_rct, pred_rct = self.ml_rct_['reaction'].scoring(rct_x,sr_rct_y)
            pred_rct.set_index(df_rct.index,drop=True,inplace=True)
            df_prd_pred = pd.concat([df_prd,pred_prd],axis=1)
            df_rct_pred = pd.concat([df_rct,pred_rct],axis=1)
            DataFramePadding(score_prd, score_rct)
            self._save_results(score_prd, score_rct, df_prd_pred, df_rct_pred, 'test')
            return score_prd, score_rct
        
        else:
            is_first = True
            for subgroup, subgroup_df in df.groupby(group_subgroup_col):
                print(f'Subset: {subgroup}')
                df_sub_prd   = subgroup_df.drop_duplicates(group_index_col)
                df_sub_rct   = subgroup_df.copy()
                sr_sub_prd_y = df_sub_prd[obj_col].copy()
                sr_sub_rct_y = to_Series(df_sub_rct,group_index_col,obj_col)
                prd_sub_x_raw, rct_sub_x_raw = self._variable_generator(
                    df_sub_prd,df_sub_rct,product_smiles_col,reactant_smiles_col,split_components)
                prd_sub_x = funcTanimotoSklearn(prd_sub_x_raw,self.prd_set_var,n_jobs=self.core)
                rct_sub_x = AverageTanimotoKernel(rct_sub_x_raw,self.rct_set_var,n_jobs=self.core)

                score_sub_prd, pred_sub_prd = self.ml_prd_[subgroup].scoring(prd_sub_x,sr_sub_prd_y)
                score_sub_rct, pred_sub_rct = self.ml_rct_[subgroup].scoring(rct_sub_x,sr_sub_rct_y)
                DataFramePadding(score_sub_prd, score_sub_rct)
                if _which == 'test':
                    if is_first:
                        self.best_subgroup_estimators_index_ = {ml : (subgroup,score_sub_rct.loc[ml, 'r2']) for ml in self.mls_rct}
                    else:
                        for ml in self.mls_rct:
                            r2 = score_sub_rct.loc[ml, 'r2']
                            if r2 > self.best_subgroup_estimators_index_[ml][1]:
                                self.best_subgroup_estimators_index_[ml] = (subgroup,r2)
                if df_whole is not None:
                    df_sub_prd_whole   = df_whole[df_whole[group_subgroup_col]==subgroup].drop_duplicates(group_index_col)
                    sr_sub_prd_whole_y = df_sub_prd_whole[obj_col].copy()
                    prd_whole_sub_x    = self._var_gen(df_sub_prd_whole,product_smiles_col,False)
                    score_sub_prd_whole, pred_sub_prd_whole = self.ml_prd_whole_[subgroup].scoring(prd_whole_sub_x,sr_sub_prd_whole_y)
                    DataFramePadding(score_sub_prd, score_sub_prd_whole)
                
                # Process results
                # Product
                score_sub_prd['Rep_reaction'] = subgroup
                score_sub_prd['template'] = self.template_[subgroup]
                score_sub_prd['data_statistics'] = str(AnalyzeSomeStatisticalValues(sr_sub_prd_y))
                score_sub_prd['num_unique_scaf'] = unique_scaf_length(df_sub_prd[product_smiles_col].to_list(),False)
                score_sub_prd.set_index(['Rep_reaction','template','num_sample','num_unique_scaf','data_statistics',score_sub_prd.index],inplace=True)
                score_prd = score_sub_prd.copy() \
                    if is_first else pd.concat([score_prd, score_sub_prd])
                df_prd_pred = pd.concat([df_sub_prd,pred_sub_prd],axis=1) \
                    if is_first else pd.concat(
                        [df_prd_pred, pd.concat([df_sub_prd,pred_sub_prd],axis=1)])
                if df_whole is not None:
                    score_sub_prd_whole['Rep_reaction'] = subgroup
                    score_sub_prd_whole['template'] = self.template_[subgroup]
                    score_sub_prd_whole['data_statistics'] = str(AnalyzeSomeStatisticalValues(sr_sub_prd_whole_y))
                    score_sub_prd_whole.set_index(['Rep_reaction','template','num_sample','data_statistics',score_sub_prd_whole.index],inplace=True)
                    score_prd_whole = score_sub_prd_whole.copy() \
                        if is_first else pd.concat([score_prd_whole, score_sub_prd_whole])
                    df_prd_pred_whole = pd.concat([df_sub_prd,pred_sub_prd_whole],axis=1) \
                        if is_first else pd.concat(
                            [df_prd_pred_whole, pd.concat([df_sub_prd,pred_sub_prd_whole],axis=1)])
                # Reactant
                score_sub_rct['Rep_reaction'] = subgroup
                score_sub_rct['template'] = self.template_[subgroup]
                score_sub_rct['data_statistics'] = str(AnalyzeSomeStatisticalValues(sr_sub_rct_y))
                score_sub_rct['num_unique_scaf'] = unique_scaf_length(df_sub_rct[reactant_smiles_col].to_list(),True)
                score_sub_rct.set_index(['Rep_reaction','template','num_sample','num_unique_scaf','data_statistics',score_sub_rct.index],inplace=True)
                pred_sub_rct.set_index(df_sub_rct.index,drop=True,inplace=True)
                score_rct = score_sub_rct.copy() \
                    if is_first else pd.concat([score_rct, score_sub_rct])
                df_rct_pred = pd.concat([df_sub_rct,pred_sub_rct],axis=1) \
                    if is_first else pd.concat(
                        [df_rct_pred, pd.concat([df_sub_rct,pred_sub_rct],axis=1)])
                is_first = False
            not_scored = [x for x in self.ml_prd_.keys() if x not in set(df[group_subgroup_col])]
            for ns in not_scored:
                print(f'Subset: {ns}')
                score_sub_prd = self.ml_prd_[ns].scoring(None,None)                
                score_sub_rct = self.ml_rct_[ns].scoring(None,None)
                DataFramePadding(score_sub_rct, score_sub_prd)
                score_sub_prd['Rep_reaction'] = ns
                score_sub_prd['template'] = self.template_[ns]
                score_sub_prd['num_unique_scaf'] = str([0])
                score_sub_prd['data_statistics'] = str(AnalyzeSomeStatisticalValues([]))
                score_sub_prd.set_index(['Rep_reaction','template','num_sample','num_unique_scaf','data_statistics',score_sub_prd.index],inplace=True)
                score_prd = pd.concat([score_prd, score_sub_prd])
                score_sub_rct['Rep_reaction'] = ns
                score_sub_rct['template'] = self.template_[ns]
                score_sub_rct['num_unique_scaf'] = str([0, 0])
                score_sub_rct['data_statistics'] = str(AnalyzeSomeStatisticalValues([]))
                score_sub_rct.set_index(['Rep_reaction','template','num_sample','num_unique_scaf','data_statistics',score_sub_rct.index],inplace=True)
                score_rct = pd.concat([score_rct, score_sub_rct])
                if df_whole is not None:
                    score_sub_prd_whole = self.ml_prd_whole_[ns].scoring(None, None)
                    DataFramePadding(score_sub_prd, score_sub_prd_whole)
                    score_sub_prd_whole['Rep_reaction'] = ns
                    score_sub_prd_whole['template'] = self.template_[ns]
                    score_sub_prd_whole['data_statistics'] = str(AnalyzeSomeStatisticalValues([]))
                    score_sub_prd_whole.set_index(['Rep_reaction','template','num_sample','data_statistics',score_sub_prd_whole.index],inplace=True)
                    score_prd_whole = pd.concat([score_prd_whole, score_sub_prd_whole])
                    
            self._save_results(score_prd, df_prd_pred, score_rct, df_rct_pred, which=f'{_which}')
            if df_whole is None:
                return score_prd, score_rct
            self._save_results(score_prd_whole, df_prd_pred_whole,which=f'{_which}_whole')
            return score_prd, score_prd_whole, score_rct

if __name__ == "__main__":
    print(1)
    # tid = pd.read_table(path, header=0,
    #                     index_col=0)

    # model = modeling()

    # tid_desc_col = tid.iloc[:,tid.columns.get_loc('Fragment2_rdkit_mol')+1:].columns.tolist()

    # train,test = model.dataset(tid, tid_desc_col, ['pot.(log,Ki)'], identify_col='ddc_id')
    # gridsearch = ['svg', 'linear-svg', 'random-forest', 'lasso']
    # est_list = []

    # # make plot
    # fig = plt.figure(figsize=(12, 10))
    # axes_1 = fig.add_subplot(2,2,1)
    # axes_2 = fig.add_subplot(2,2,2)
    # axes_3 = fig.add_subplot(2,2,3)
    # axes_4 = fig.add_subplot(2,2,4)
    # axes = [axes_1, axes_2, axes_3, axes_4]
    # df_pred_train_fr = train.copy().iloc[:,:4]
    # df_pred_test_fr = test.copy().iloc[:,:4]

    # for i, m in enumerate(gridsearch):
    #     model.set_models(m)
    #     model.best_param_search(save_prefix=f'{name}_{m}')
    #     pred_y_train, pred_y_test = model.predict()
    #     err_y_train = np.abs(np.array(df_pred_train_fr.loc[:,'pot.(log,Ki)'].to_list()) - pred_y_train.reshape(-1))
    #     err_y_test = np.abs(np.array(df_pred_test_fr.loc[:,'pot.(log,Ki)'].to_list()) - pred_y_test.reshape(-1))
    #     df_pred_train_fr = pd.concat([df_pred_train_fr.reset_index().drop('index',axis=1), pd.DataFrame(pred_y_train, columns=[f'pred_{m}']), pd.DataFrame(err_y_train, columns=[f'abs_err_{m}'])], axis=1)
    #     df_pred_test_fr = pd.concat([df_pred_test_fr.reset_index().drop('index',axis=1), pd.DataFrame(pred_y_test, columns=[f'pred_{m}']), pd.DataFrame(err_y_test, columns=[f'abs_err_{m}'])], axis=1)
    #     est_list.append(model.evaluation())
    #     model.plot(axes[i])
    
    # fig.suptitle(f'{name}')
    # fig.tight_layout()
    # fig.savefig(f'{name}.png')

    # model.importance(f'{name}')

    # # make table of values
    # val = pd.DataFrame({"Name":['R2','R2pred','RMSE','RMSEpred'],
    #                     "Gausian-SVM":est_list[0],
    #                     "Linear-SVM":est_list[1],
    #                     "Random-forest":est_list[2],
    #                     "Lasso":est_list[3]
    #                     })
    # print(val)