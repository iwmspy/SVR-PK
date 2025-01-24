# wrapped kernel modules

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import check_pairwise_arrays
from joblib import Parallel,delayed,wrap_non_picklable_objects

@delayed
@wrap_non_picklable_objects
def _TaniSim_vec(X,y):
    num = np.asarray(X.minimum(y).sum(1)).reshape(-1,1)
    denom = np.asarray(X.maximum(y).sum(1)).reshape(-1,1)
    return num / denom

def _simcalc(X,y):
    num = np.asarray(X.minimum(y).sum(1)).reshape(-1,1)
    denom = np.asarray(X.maximum(y).sum(1)).reshape(-1,1)
    return num / denom

def SparseTanimotoSimilarity(X, Y=None,n_jobs=1):
    """
    X: scipy.sparse CSR matrix, shape (m1, n)
    Y: scipy.sparse CSR matrix, shape (m2, n)
    n_jobs: int, number of threads, if 1 is inputted, Parallelization will not work (only single thread).
    returns: similarity matrix between X and Y, shape (m1, m2)
    """
    X,Y = check_pairwise_arrays(X,Y,dtype=bool)
    m1 = X.shape[0]
    m2 = Y.shape[0]
    d = []
    for _ in range(m2):
        d.append(_simcalc(X,Y[np.repeat(_, m1)]))
    return np.hstack(d)

def TanimotoSimilarity(mat_x,mat_y=None,n_jobs=1):
    """
    inputs
    -------
    X : ndarray of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_features)
        Array of pairwise distances between samples, or a feature array.

    Y : ndarray of shape (n_samples_Y, n_features), default=None
        An optional second feature array. 

    Returns
    -------
    D : ndarray of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_samples_Y)
        A distance matrix D such that D_{i, j} is the distance between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the distance between the ith array
        from X and the jth array from Y.
    """
    return 1-pairwise_distances(mat_x.astype(bool, copy=False),
                                mat_y.astype(bool, copy=False),
                                metric='jaccard',n_jobs=n_jobs)

def SparseJaccardDistance(mat_x,mat_y=None,n_jobs=1):
    """
    inputs
    -------
    X : ndarray of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_features)
        Array of pairwise distances between samples, or a feature array.

    Y : ndarray of shape (n_samples_Y, n_features), default=None
        An optional second feature array. 

    Returns
    -------
    D : ndarray of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_samples_Y)
        A distance matrix D such that D_{i, j} is the distance between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the distance between the ith array
        from X and the jth array from Y.
    """
    return 1 - SparseTanimotoSimilarity(mat_x,mat_y,n_jobs=n_jobs)

def funcTanimotoSklearn(x, y, n_jobs=1):
    x, y = check_pairwise_arrays(x, y)
    if (x.ndim == 1) and (y.ndim ==1):
        jdist = pairwise_distances(x.astype(bool, copy=False).reshape(1,-1), 
                                   y.astype(bool, copy=False).reshape(1,-1), 
                                   metric='jaccard',n_jobs=n_jobs)
    else:
        jdist = pairwise_distances(x.astype(bool, copy=False), 
                                   y.astype(bool, copy=False), 
                                   metric='jaccard',n_jobs=n_jobs)
    return 1 - jdist

def ProductTanimotoKernel(x, y=None, len_first=8192, n_jobs=1):
    x, y = check_pairwise_arrays(x, y)
    x1, x2 = x[:,:len_first], x[:,len_first:]
    y1, y2 = y[:,:len_first], y[:,len_first:]
    k1 = funcTanimotoSklearn(x1, y1, n_jobs=n_jobs)
    k2 = funcTanimotoSklearn(x2, y2, n_jobs=n_jobs)
    kproduct = k1*k2
    return kproduct

def SparseProductTanimotoKernel(x, y=None, len_first=8192, n_jobs=1):
    x, y = check_pairwise_arrays(x, y)
    x1, x2 = x[:,:len_first], x[:,len_first:]
    y1, y2 = y[:,:len_first], y[:,len_first:]
    k1 = SparseTanimotoSimilarity(x1,y1,n_jobs=n_jobs)
    k2 = SparseTanimotoSimilarity(x2,y2,n_jobs=n_jobs)
    kproduct = k1*k2
    return kproduct

if __name__=='__main__':
    print(':)')