# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""
import os
import sys
if __name__=='__main__':
    pwd = os.path.dirname(os.path.abspath(os.path.join(__file__,'../')))
    sys.path.append(pwd)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# import sample_functions

def k3n_error(x_1, x_2, k):
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
        _replace_zero_with_the_smallest_positive_values(x_2_distance[i, :])

    identity_matrix = np.eye(len(x_1_distance), dtype=bool)
    knn_distance_in_x_1 = np.sort(x_2_distance[:, x_1_sorted_indexes[:, 1:k + 1]][identity_matrix])
    knn_distance_in_x_2 = np.sort(x_2_distance)[:, 1:k + 1]

    sum_k3n_error = (
            (knn_distance_in_x_1 - knn_distance_in_x_2) / knn_distance_in_x_2
    ).sum()
    return sum_k3n_error / x_1.shape[0] / k

def _replace_zero_with_the_smallest_positive_values(arr):
    """
    Replace zeros in array with the smallest positive values.

    Parameters
    ----------
    arr: numpy.array
    """
    arr[arr == 0] = np.min(arr[arr != 0])

def scaler(df:pd.DataFrame):
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

def tSNE_Jaccard(dataset:np.array, label:list=None, n_components=2, perplexity=30):
    n_samples, n_features = dataset.shape
    dim = min(n_samples, n_features)
    assert(dim>=n_components)
    t_mod  = TSNE(perplexity=perplexity, 
                  n_components=n_components, 
                  init='pca', 
                  metric='jaccard',
                  random_state=0,
                  n_jobs=-1)
    res    = t_mod.fit_transform(dataset.astype(bool))
    res_df = pd.DataFrame(res, columns=[f'comp_{i}' for i in range(1, n_components+1, 1)]) 
    if label!=None: res_df['label'] = label
    return t_mod, res_df

def tSNE(dataset:pd.DataFrame, label_col=None, scale=True,comp=2):
    if label_col!=None:
        dataset_ = dataset.copy()
        dataset.drop(label_col,axis=1,inplace=True)
    n_samples,n_features = dataset.shape
    dim = min(n_samples,n_features)
    if dim<2:
        print("This dataset doesn't have enough data dimension. Quit.")
        return
    start = 1
    end = dim - 1
    sp = 20
    interval = (end-start)/(sp-1)
    candidates_of_perplexity = np.array([int(start+i*interval) for i in range(sp)])
    k_in_k3n_error = 10

    if scale:
        autoscaled_dataset = scaler(dataset)  # Scaling
    else:
        autoscaled_dataset = dataset

    # Optimalization perplexity by k3n-error 
    k3n_errors = []
    for index, perplexity in tqdm(enumerate(candidates_of_perplexity), total=len(candidates_of_perplexity)):
        if perplexity >= autoscaled_dataset.shape[0]:
            perplexity = candidates_of_perplexity[index-1]
            candidates_of_perplexity = candidates_of_perplexity[:index]
            print(f"Perplexity is over num_of_features. Perplexity was set {perplexity}.")
            break
        # print(index + 1, '/', len(candidates_of_perplexity))
        t = TSNE(perplexity=perplexity, n_components=comp, init='pca', random_state=0,
                 n_jobs=-1).fit_transform(autoscaled_dataset)
        scaled_t = (t - t.mean(axis=0)) / t.std(axis=0, ddof=1)

        k3n_errors.append(
            k3n_error(autoscaled_dataset, scaled_t, k_in_k3n_error) + k3n_error(
                scaled_t, autoscaled_dataset, k_in_k3n_error))
    fig_tSNE = plt.figure(figsize=(21,11))
    axes_1 = fig_tSNE.add_subplot(1,2,1)
    axes_2 = fig_tSNE.add_subplot(1,2,2)

    axes_2.scatter(candidates_of_perplexity, k3n_errors, c='blue')
    axes_2.set_aspect(1.0/axes_2.get_data_ratio(), adjustable='box')
    axes_2.set_xlabel("perplexity")
    axes_2.set_ylabel("k3n-errors")
    axes_2.set_title("Optimalization perplexity by k3n-error")
    axes_2.legend()
    # plt.show()
    optimal_perplexity = candidates_of_perplexity[np.where(k3n_errors == np.min(k3n_errors))[0][0]]
    print('Optimal value of perplexity by k3n-error :', optimal_perplexity)

    # t-SNE
    t = TSNE(perplexity=optimal_perplexity, n_components=comp, init='pca', random_state=0,n_jobs=-1).fit_transform(autoscaled_dataset)
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

def run_pca(dataset:pd.DataFrame, label_col=None, scale=True):
    if label_col!=None:
        dataset_ = dataset.copy()
        dataset.drop(label_col,axis=1,inplace=True)
    n_samples,n_features = dataset.shape
    dim = min(n_samples,n_features)
    if dim<2:
        print("This dataset doesn't have enough data dimension. Quit.")
        return
    if scale:
        autoscaled_dataset = scaler(dataset)  # Scaling
    else:
        autoscaled_dataset = dataset.to_numpy()
    pca = PCA(n_components=2)
    p = pd.DataFrame(pca.fit_transform(autoscaled_dataset), 
                     index=dataset.index, columns=['comp_1', 'comp_2']) 
    ax1, ax2 = pca.explained_variance_ratio_
    if label_col!=None:
        p=pd.concat([p,dataset_[label_col]],axis=1)
    fig_pca = plt.figure(figsize=(11,11))
    axes = fig_pca.add_subplot(1,1,1)
    if label_col!=None:
        for label in dataset_[label_col].unique().tolist():
            p_select = p[p[label_col]==label]
            axes.scatter(p_select.iloc[:, 0], p_select.iloc[:, 1],
                         label=label)
    else:
        axes.scatter(p.iloc[:, 0], p.iloc[:, 1], c='blue')
    axes.set_aspect(1.0/axes.get_data_ratio(), adjustable='box')
    axes.set_xlabel(f'comp_1 (exp_ratio : {ax1})')
    axes.set_ylabel(f'comp_2 (exp_ratio : {ax2})')
    axes.set_title('PCA scatter')
    axes.legend()

    fig_pca.tight_layout()
    return p, fig_pca

if __name__=='__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    dataset = pd.read_csv('iris.csv', index_col=0, header=0)
    # t,fig = tSNE(dataset,label_col='species')
    p,fig = run_pca(dataset,label_col='species')

    fig.show()
