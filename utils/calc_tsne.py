# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from tqdm import tqdm
from joblib import Parallel,delayed

# import sample_functions
class calc_tsne:
    def __init__(self,n_jobs=-1):
        self.n_jobs = n_jobs

    def k3n_error(self,x_1, x_2, k):
        """
        k-nearest neighbor normalized error (k3n-error)

        When X1 is data of X-variables and X2 is data of Z-variables
        (low-dimensional data), this is k3n error in visualization (k3n-Z-error).
        When X1 is Z-variables (low-dimensional data) and X2 is data of data of
        X-variables, this is k3n error in reconstruction (k3n-X-error).

        k3n-error = k3n-Z-error + k3n-X-error

        Parameters
        ----------
        x_1: numpy.array or pandas.DataFrame
        x_2: numpy.array or pandas.DataFrame
        k: int
            The numbers of neighbor

        Returns
        -------
        k3n_error : float
            k3n-Z-error or k3n-X-error
        """
        x_1 = np.array(x_1)
        x_2 = np.array(x_2)

        x_1_distance = cdist(x_1, x_1)
        x_1_sorted_indexes = np.argsort(x_1_distance, axis=1)
        x_2_distance = cdist(x_2, x_2)

        for i in range(x_2.shape[0]):
            self._replace_zero_with_the_smallest_positive_values(x_2_distance[i, :])

        identity_matrix = np.eye(len(x_1_distance), dtype=bool)
        knn_distance_in_x_1 = np.sort(x_2_distance[:, x_1_sorted_indexes[:, 1:k + 1]][identity_matrix])
        knn_distance_in_x_2 = np.sort(x_2_distance)[:, 1:k + 1]

        sum_k3n_error = (
                (knn_distance_in_x_1 - knn_distance_in_x_2) / knn_distance_in_x_2
        ).sum()
        return sum_k3n_error / x_1.shape[0] / k

    def _replace_zero_with_the_smallest_positive_values(self,arr):
        """
        Replace zeros in array with the smallest positive values.

        Parameters
        ----------
        arr: numpy.array
        """
        arr[arr == 0] = np.min(arr[arr != 0])

    def scaler(self,df:pd.DataFrame):
        df = df.loc[:,(df != 0).any(axis=0)]
        new_df = pd.DataFrame(index=df.index)
        for col in df.columns:
            if df[col].std() > 0:
                new_df = pd.concat([new_df, (df[col]-df[col].mean())/df[col].std()],axis=1)
            else:
                continue
                # new_df = pd.concat([new_df, df[col]],axis=1)
        new_df.sort_index(axis=1,inplace=True)
        return new_df

    def process_perplexity(self, perplexity):
        # if perplexity >= self.autoscaled_dataset.shape[0]:
        #     perplexity = candidates_of_perplexity[index-1]
        #     candidates_of_perplexity = candidates_of_perplexity[:index]
        #     print(f"Perplexity is over num_of_features. Perplexity was set {perplexity}.")
        #     return None

        t = TSNE(perplexity=perplexity, n_components=self.comp, init='pca', random_state=0,
                 n_jobs=self.n_jobs
                 ).fit_transform(self.autoscaled_dataset)
        scaled_t = (t - t.mean(axis=0)) / t.std(axis=0, ddof=1)

        return self.k3n_error(self.autoscaled_dataset, scaled_t, self.k_in_k3n_error) + self.k3n_error(scaled_t, self.autoscaled_dataset, self.k_in_k3n_error)

    def tSNE(self,dataset:pd.DataFrame, label_col=None, scale=True, comp=2, split=20):
        self.comp = comp
        n_samples,n_features = dataset.shape
        dim = min(n_samples,n_features)
        if dim<2:
            print("This dataset doesn't have enough data dimension. Quit.")
            return
        start = 1
        end = dim - 1
        interval = (end-start)/(split-1)
        self.candidates_of_perplexity = np.array([int(start+i*interval) for i in range(split)])
        self.k_in_k3n_error = 10

        # candidates_of_perplexity = np.arange(5, 105, 5, dtype=int)

        # dataset = pd.read_csv('iris_without_species.csv', index_col=0)
        if label_col!=None:
            dataset_ = dataset.copy()
            dataset.drop(label_col,axis=1,inplace=True)
        if scale:
            self.autoscaled_dataset = self.scaler(dataset)  # Scaling
        else:
            self.autoscaled_dataset = dataset

        # Optimalization perplexity by k3n-error 
        k3n_errors = []
        # for index, perplexity in tqdm(enumerate(self.candidates_of_perplexity), total=len(self.candidates_of_perplexity)):
        #     # if perplexity >= self.autoscaled_dataset.shape[0]:
        #     #     perplexity = self.candidates_of_perplexity[index-1]
        #     #     self.candidates_of_perplexity = self.candidates_of_perplexity[:index]
        #     #     print(f"Perplexity is over num_of_features. Perplexity was set {perplexity}.")
        #     #     break
        #     # print(index + 1, '/', len(candidates_of_perplexity))
        #     t = TSNE(perplexity=perplexity, n_components=comp, init='pca', random_state=0).fit_transform(self.autoscaled_dataset)
        #     scaled_t = (t - t.mean(axis=0)) / t.std(axis=0, ddof=1)

        #     k3n_errors.append(
        #         self.k3n_error(self.autoscaled_dataset, scaled_t, self.k_in_k3n_error) + self.k3n_error(
        #             scaled_t, self.autoscaled_dataset, self.k_in_k3n_error))
        # values_to_process = [(index, perplexity) for index, perplexity in enumerate(self.candidates_of_perplexity)]
        values_to_process = [perplexity for perplexity in self.candidates_of_perplexity]

        results = Parallel(n_jobs=self.n_jobs)(delayed(self.process_perplexity)(
            params) for params in values_to_process if params is not None)

        k3n_errors = [result for result in results if result is not None]

        fig_tSNE = plt.figure(figsize=(21,11))
        axes_1 = fig_tSNE.add_subplot(1,2,1)
        axes_2 = fig_tSNE.add_subplot(1,2,2)

        axes_2.scatter(self.candidates_of_perplexity, k3n_errors, c='blue')
        axes_2.set_aspect(1.0/axes_2.get_data_ratio(), adjustable='box')
        axes_2.set_xlabel("perplexity")
        axes_2.set_ylabel("k3n-errors")
        axes_2.set_title("Optimalization perplexity by k3n-error")
        axes_2.legend()
        # plt.show()
        optimal_perplexity = self.candidates_of_perplexity[np.where(k3n_errors == np.min(k3n_errors))[0][0]]
        print('Optimal value of perplexity by k3n-error :', optimal_perplexity)

        # t-SNE
        t = TSNE(perplexity=optimal_perplexity, 
                 n_components=comp, 
                 init='pca', 
                 random_state=0,
                 n_jobs=self.n_jobs).fit_transform(self.autoscaled_dataset)
        t = pd.DataFrame(t, index=dataset.index, columns=['t_1', 't_2']) 
        if label_col!=None:
            t=pd.concat([t,dataset_[label_col]],axis=1)

        # scatter
        if label_col!=None:
            for label in dataset_[label_col].unique().tolist():
                t_select = t[t[label_col]==label]
                axes_1.scatter(t_select.iloc[:, 0], t_select.iloc[:, 1],label=label)
        else:
            axes_1.scatter(t.iloc[:, 0], t.iloc[:, 1], c='blue')
        axes_1.set_aspect(1.0/axes_1.get_data_ratio(), adjustable='box')
        axes_1.set_xlabel('$t_1$')
        axes_1.set_ylabel('$t_2$')
        axes_1.set_title('t-SNE scatter')
        axes_1.legend()

        fig_tSNE.tight_layout()

        return t, fig_tSNE

if __name__=='__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    dataset = pd.read_csv('iris.csv', index_col=0, header=0)

    sn = calc_tsne()

    t,fig = sn.tSNE(dataset,label_col='species')

    fig.show()
