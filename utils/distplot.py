'''
'''

import pandas as pd
import numpy as np
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, Recap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch

class dist:
    def __init__(self, x1:pd.DataFrame, x2:pd.DataFrame, method='euclidean', scale=False, n_jobs=1):
        '''
        method : calculation method, you can choose 'euclidean' or 'tanimoto'.
            'euclidean' returns distances calculated by euclidean distance.
            'tanimoto' returns distances calculated by tanimoto similarity (must be binary).
        '''
        self.method = method
        assert self.method in ('euclidean','tanimoto'),\
            'You must choose which "euclidean" or "tanimoto".'
        self.n_jobs = n_jobs
        self.x1_idx = x1.index
        self.x2_idx = x2.index
        if (method == 'euclidean') and scale:
            ss_x1 = StandardScaler()
            self.x1 = ss_x1.fit_transform(x1)
            ss_x2 = StandardScaler()
            self.x2 = ss_x2.fit_transform(x2)
        else:
            if method == 'euclidean':
                self.x1 = np.array(x1).astype(np.float32)
                self.x2 = np.array(x2).astype(np.float32)
            else:
                self.x1 = np.array(x1).astype(bool)
                self.x2 = np.array(x2).astype(bool)

    def calcdist(self):
        if self.method == 'euclidean':
            self.x1_dist = pairwise_distances(self.x1, metric='euclidean',n_jobs=self.n_jobs)
            self.x2_dist = pairwise_distances(self.x2, metric='euclidean',n_jobs=self.n_jobs)
        else:
            self.x1_dist = pairwise_distances(self.x1, metric='jaccard',n_jobs=self.n_jobs)
            self.x2_dist = pairwise_distances(self.x2, metric='jaccard',n_jobs=self.n_jobs)
    
    def ret(self, return_df=False):
        # return self.dist_list
        if return_df:
            df_x1_dist = pd.DataFrame(self.x1_dist,index=self.x1_idx,columns=self.x1_idx)
            df_x2_dist = pd.DataFrame(self.x2_dist,index=self.x2_idx,columns=self.x2_idx)
            return df_x1_dist, df_x2_dist
        return self.x1_dist, self.x2_dist
    
    def distplot(self, title='distance_plot', axes_name_x='x1', axes_name_y='x2', save_name='distance_plot.png'):
        fig = plt.figure(figsize=(12, 10))
        axes = fig.add_subplot(1,1,1)
        if self.x1_dist.shape != self.x2_dist.shape:
            raise ValueError('Shape of df must be same')
        for idx in tqdm(range(self.x1_dist.shape[0])):
            axes.scatter(self.x1_dist[idx][idx:], self.x2_dist[idx][idx:])
        lim = [np.min([axes.get_xlim(), axes.get_ylim()]),
               np.max([axes.get_xlim(), axes.get_ylim()])]
        axes.plot(lim, lim, color='red')
        axes.set_xlim(lim)
        axes.set_ylim(lim)
        axes.set_aspect(1.0/axes.get_data_ratio(), adjustable='box')
        axes.set_title(f'{title}',fontsize=16)
        axes.set_xlabel(f"{axes_name_x}",fontsize=16)
        axes.set_ylabel(f"{axes_name_y}",fontsize=16)
        axes.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close(fig)

class dist_torch(dist):
    def __init__(self, x1:pd.DataFrame, x2:pd.DataFrame, method='euclidean', scale=False):
        '''
        method : calculation method, you can choose 'euclidean' or 'tanimoto'.
            'euclidean' returns distances calculated by euclidean distance.
            'tanimoto' returns distances calculated by tanimoto similarity (must be binary).
        '''
        super().__init__(x1=x1,x2=x2,method=method,scale=scale)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if method == 'euclidean':
            self.x1_tsr = torch.tensor(self.x1,dtype=torch.float32,device=dev)
            self.x2_tsr = torch.tensor(self.x2,dtype=torch.float32,device=dev)
        else:
            self.x1_tsr = torch.tensor(self.x1,dtype=torch.bool,device=dev)
            self.x2_tsr = torch.tensor(self.x2,dtype=torch.bool,device=dev)
    def jaccard_distance_matrix(self,tensor):
        num_samples = tensor.size(0)
        distance_matrix = torch.zeros((num_samples, num_samples))
        
        for i in range(num_samples):
            for j in range(num_samples):
                intersection = torch.logical_and(tensor[i], tensor[j]).sum()
                union = torch.logical_or(tensor[i], tensor[j]).sum()
                jaccard_coefficient = intersection.float() / union.float()
                jaccard_dist = 1.0 - jaccard_coefficient
                distance_matrix[i][j] = jaccard_dist
        
        return distance_matrix
    
    def calcdist(self):
        if self.method == 'euclidean':
            x1_dist_tsr = torch.cdist(self.x1_tsr, self.x1_tsr, p=2)
            x2_dist_tsr = torch.cdist(self.x2_tsr, self.x2_tsr, p=2)
        else:
            x1_dist_tsr = self.jaccard_distance_matrix(self.x1_tsr)
            x2_dist_tsr = self.jaccard_distance_matrix(self.x2_tsr)
        self.x1_dist = x1_dist_tsr.cpu().numpy() \
            if torch.cuda.is_available() else x1_dist_tsr.numpy()
        self.x2_dist = x2_dist_tsr.numpy() \
            if torch.cuda.is_available() else x2_dist_tsr.numpy()

class dist_multidim(dist):
    def calcdist(self):
        if self.method == 'euclidean':
            self.multidimdist = pairwise_distances(self.x1, self.x2, metric='euclidean',n_jobs=self.n_jobs)
        else:
            self.multidimdist = pairwise_distances(self.x1, self.x2, metric='jaccard',n_jobs=self.n_jobs)
        return self.multidimdist
    
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

if __name__=='__main__':
    data1 = pd.DataFrame([[1, 1, 0, 1, 0, 0],
                          [1, 0, 1, 1, 0, 1]])
    data2 = pd.DataFrame([[0, 1, 1, 0, 1],
                          [1, 0, 1, 0, 0],])
    data3 = pd.DataFrame([[0, 1, 1],
                          [1, 0, 1]])
    data4 = pd.DataFrame([[0, 3, 1, 0, 5],
                          [2, 0, 4, 3, 0]])
    data5 = pd.DataFrame([[0, 1, 1, 0, 1, 0],
                          [1, 0, 1, 0, 0, 1],
                          [1, 1, 1, 1, 0, 1],
                          [0, 1, 1, 1, 0, 0]])
    
    dis_binary = dist_torch(data1, data2, method='tanimoto')
    dis_int = dist_torch(data1, data4,scale=True)
    # dis_binary = dist(data1, data2, method='tanimoto')
    # dis_int = dist(data1, data4,scale=True)
    # dis = dist(data1, data3) # might raise error
    dis_mdm = dist_multidim(data1, data5, method="tanimoto", n_jobs=-1)
    lis_mdm = dis_mdm.calcdist()
    df_sims = spritvectorsims(data5, data1, data5.columns[:3],data5.columns[3:])
    dis_binary.calcdist()
    dis_int.calcdist()
    lis_binary_1, lis_binary_2 = dis_binary.ret(return_df=True)
    lis_int_1, lis_int_2 = dis_int.ret(return_df=True)

    dis_binary.distplot()
