import os
from time import time

import numpy as np
import pandas as pd
from rdkit.DataStructs import TanimotoSimilarity,CreateFromBitString
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from tqdm import tqdm

from models._kernel_and_mod import funcJaccardSklearn, funcTanimotoSklearn, ProductJaccardKernel
from utils.utility import mean_of_top_n_elements
from utils.chemutils import MorganbitCalcAsVectors


split = 10000
cpus  = os.cpu_count()

def sim_jaccard_from_sparse_list(l1:np.ndarray,l2:np.ndarray):
    return np.intersect1d(l1,l2).shape[0] / np.union1d(l1,l2).shape[0]

class neighbors:
    def __init__(self,bits_array,n_neigh=3,jobs=-1):
        self.nn = NearestNeighbors(n_neighbors=n_neigh,
                              metric='jaccard',
                              n_jobs=jobs)
        self.bits_array_train = bits_array.astype(bool,copy=False)
        self.nn.fit(self.bits_array_train)
    
    def calcneighs(self,bits_list):
        dists, inds = self.nn.kneighbors([np.array(bits_list).astype(bool, copy=False)])
        return dists, inds
    
    def calcsim(self,bits_list):
        dists, inds = self.calcneighs(bits_list)
        sims = []
        for ind in inds[0]:
            bits_list_ = self.bits_array_train[ind]
            bits_ = ''.join(str(bit) for bit in bits_list_)
            bits = ''.join(str(bit) for bit in bits_list)
            fp1 = CreateFromBitString(bits_)
            fp2 = CreateFromBitString(bits)
            sim = TanimotoSimilarity(fp1,fp2)
            sims.append(sim)
        return np.mean(np.array(sims))
    
    def calcsims_(self,bits_array):
        sims = []
        for i, bits_list in enumerate(tqdm(bits_array)):
            sim = self.calcsim(bits_list)
            sims.append(sim)
        return sims
    
    def calcsims(self,bits_array_,split=10000):
        sims = []
        idxs = []
        if split:
            bits_array_list = np.array_split(bits_array_, cpus)
        else:
            bits_array_list = [bits_array_]
        print(f'---Start similarity calculation--- \n Method will be iterated {len(bits_array_list)} time(s).')
        for iter, bits_array in enumerate(bits_array_list):
            print(f'***Iterate {iter+1}***')
            start = time()
            dists, idx = self.nn.kneighbors(bits_array.astype(bool, copy=False))
            print(f'Took {time()-start} seconds for identifing neighbors.')
            similars = 1 - dists
            similars_mean = np.mean(similars,axis=1).tolist()
            idxs.extend(idx)
            sims.extend(similars_mean)
            print(f'***Iterate {iter+1} end. Took {time()-start} seconds.***')
        return sims
    
class neighbors_sparse:
    def __init__(self,train,n_neigh=3,n_jobs=-1,*keys):
        self.train = train.astype(bool,copy=False)
        self.n_neigh = n_neigh
        self.n_jobs = n_jobs
    
    def transform(self,bits_sp):
        self.kernel = funcJaccardSklearn(self.train,bits_sp,n_jobs=self.n_jobs)
        top_n_indices = np.argsort(-self.kernel.T,axis=1)[:,:self.n_neigh]
        sim_val = np.sort(self.kernel.T,axis=1)[:,::-1][:,:self.n_neigh]
        return top_n_indices,sim_val


class NearestNeighborSearchFromSmiles:
    def __init__(self,n_neigh=1,radius=2,bit_len=8192,n_jobs=-1,split_components=False) -> None:
        self.n_n = n_neigh
        self.rad = radius
        self.bl  = bit_len
        self.job = n_jobs
        self.spc = split_components
        metric   = ProductJaccardKernel if self.spc else funcJaccardSklearn
        self.nn  = NearestNeighbors(
            n_neighbors=self.n_n,
            metric=metric,
            n_jobs=self.job)
        
    def fit(self,smiles_list=None,bits_array=None,precalc=False):
        if not precalc:
            self.train_smiles_list = smiles_list
        self.train_bits_array  = csr_matrix(np.array(
            MorganbitCalcAsVectors(smiles_list,self.rad,self.bl,n_jobs=self.job,split_components=self.spc))
            ) if not precalc else bits_array
        self.nn.fit(self.train_bits_array)
    
    def transform(self,smiles_list=None,bits_array=None,precalc=False,ret_indices=True):
        if not precalc:
            self.test_smiles_list = smiles_list
        self.test_bits_array  = csr_matrix(np.array(
            MorganbitCalcAsVectors(smiles_list,self.rad,self.bl,n_jobs=self.job,split_components=self.spc))
            ) if not precalc else bits_array
        dists, inds = self.nn.kneighbors(self.test_bits_array)
        inds_to_smi = np.array([[self.train_smiles_list[i] for i in ind_mod] for ind_mod in inds])
        if ret_indices: return dists, inds, inds_to_smi
        return dists, inds_to_smi


def BulkNeighborsSimilarity(train_and_test_bits_array_list,n_neigh=3,jobs=-1):
    '''
    <shape of train_and_test_bits_array_list>
        [[train, test], ....]
        each train and test datas must be np.array
    '''
    bulksims = []
    for train, test in train_and_test_bits_array_list:
        nn = neighbors(train,n_neigh=n_neigh,jobs=jobs)
        sims = nn.calcsims(test)
        bulksims.append(sims)
    bulksims = np.array(bulksims)
    return bulksims.T.tolist()

def NeighborsSimilarityForSparseMat(train_and_test_bits_array_list,n_neigh=3,n_jobs=-1):
    '''
    <shape of train_and_test_bits_array_list>
        [[train, test], ....]
        each train and test datas must be np.array
    '''
    bulksims = []
    for train, test in train_and_test_bits_array_list:
        nn = funcTanimotoSklearn(train,test,n_jobs=n_jobs)
        sims = mean_of_top_n_elements(nn.T,n_neigh).reshape(-1)
        bulksims.append(sims)
    bulksims = np.array(bulksims)
    return bulksims.T.tolist()

def uniandpairs(df_bits_1:pd.DataFrame, df_bits_2:pd.DataFrame):
    uni_dict = {}
    and_dict = {}
    for idx_1, vec_1 in df_bits_1.iterrows():
        vec_1_np = vec_1.to_numpy()
        for idx_2, vec_2 in df_bits_2.iterrows():
            vec_2_np = vec_2.to_numpy()
            if not(f"{idx_1}&{idx_2}" in uni_dict):
                uni_dict[f"{idx_1}&{idx_2}"] = np.sum(vec_1_np | vec_2_np)
                and_dict[f"{idx_1}&{idx_2}"] = np.sum(vec_1_np & vec_2_np)
            else:
                raise Exception("Same index was detected. This module does not support same indexes.")
    return uni_dict, and_dict

def spritvectorsims(df_train:pd.DataFrame, df_test:pd.DataFrame, col1:list, col2:list):
    df_train_col1, df_train_col2 = df_train.loc[:,col1], df_train.loc[:,col2]
    df_test_col1, df_test_col2 = df_test.loc[:,col1], df_test.loc[:,col2]
    uni_1, and_1 = uniandpairs(df_test_col1, df_train_col1)
    uni_2, and_2 = uniandpairs(df_test_col2, df_train_col2)
    df_sims = pd.DataFrame(index=df_test.index,columns=df_train.index)
    for test in df_sims.index:
        for train in df_sims.columns:
            pair = f'{test}&{train}'
            df_sims.loc[test, train] = (and_1[pair]+and_2[pair])/(uni_1[pair]+uni_2[pair])
    return df_sims


if __name__=="__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    nn = neighbors()
    print(1)