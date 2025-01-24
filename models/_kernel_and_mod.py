from copy import deepcopy
from sklearn.linear_model import LassoCV
import numpy as np
from scipy.sparse import isspmatrix
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import check_pairwise_arrays
from joblib import parallel_backend,Parallel,delayed
from rdkit.Chem.Fraggle import FraggleSim

class lassocv_integrate_grid(LassoCV):
    def __init__(self, *, 
                 eps = 0.001, n_alphas = 100, 
                 alphas = None, fit_intercept = True, 
                 precompute = "auto", max_iter = 1000, 
                 tol = 0.0001, copy_X = True, 
                 cv = None, verbose = False, n_jobs = None, 
                 positive = False, random_state = None, selection = "cyclic") -> None:
        super().__init__(eps=eps, n_alphas=n_alphas, alphas=alphas, 
                         fit_intercept=fit_intercept, precompute=precompute,
                         max_iter=max_iter, tol=tol, copy_X=copy_X, cv=cv, 
                         verbose=verbose, n_jobs=n_jobs, positive=positive, 
                         random_state=random_state, selection=selection)
    def fit(self, X, y, sample_weight = None):
        self.best_estimator_ = super().fit(X, y, sample_weight)
        

def TanimotoInrow(X,y):
    num = np.asarray(X.minimum(y).sum(1)).reshape(-1,1)
    denom = np.asarray(X.maximum(y).sum(1)).reshape(-1,1)
    return num / denom


def _SparseTanimotoSimilarities(X, Y=None,n_jobs=1):
    """
    X: scipy.sparse CSR matrix, shape (m1, n)
    Y: scipy.sparse CSR matrix, shape (m2, n)
    n_jobs: int, number of threads, if 1 is inputted, Parallelization will not work (only single thread).
    returns: similarity matrix between X and Y, shape (m1, m2)
    """
    m1 = X.shape[0]
    m2 = Y.shape[0]
    d = Parallel(n_jobs=n_jobs,backend='threading')(
        delayed(TanimotoInrow)(X,Y[np.repeat(_, m1)]) for _ in range(m2))
    return np.hstack(d)


# def _SparseTanimotoSimilarities(X, Y=None,n_jobs=1):
#     """
#     X: scipy.sparse CSR matrix, shape (m1, n)
#     Y: scipy.sparse CSR matrix, shape (m2, n)
#     n_jobs: int, number of threads, if 1 is inputted, Parallelization will not work (only single thread).
#     returns: similarity matrix between X and Y, shape (m1, m2)
#     """
#     m1 = X.shape[0]
#     m2 = Y.shape[0]
#     d = []
#     for _ in range(m2):
#         d.append(TanimotoInrow(X,Y[np.repeat(_, m1)]))
#     return np.hstack(d)


def TanimotoSimilarities(mat_x,mat_y=None,n_jobs=1):
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
        A similarity matrix D such that D_{i, j} is the similarity between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the similarity between the ith array
        from X and the jth array from Y.
    """
    mat_x, mat_y = check_pairwise_arrays(mat_x, mat_y)
    if isspmatrix(mat_x): return _SparseTanimotoSimilarities(mat_x,mat_y,n_jobs=n_jobs)
    return 1 - pairwise_distances(mat_x.astype(bool, copy=False),
                                mat_y.astype(bool, copy=False),
                                metric='jaccard',n_jobs=n_jobs)


def SparseJaccardDistances(mat_x,mat_y=None,n_jobs=1):
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
    mat_x, mat_y = check_pairwise_arrays(mat_x, mat_y)
    return 1 - _SparseTanimotoSimilarities(mat_x,mat_y,n_jobs=n_jobs)


def funcTanimotoSklearn(x, y=None, n_jobs=1):
    x, y  = check_pairwise_arrays(x, y)
    if isspmatrix(x): return _SparseTanimotoSimilarities(x, y)
    if (x.ndim == 1) and (y.ndim ==1):
        jdist = pairwise_distances(x.astype(bool, copy=False).reshape(1,-1), 
                                y.astype(bool, copy=False).reshape(1,-1), 
                                metric='jaccard',n_jobs=n_jobs)
    else:
        jdist = pairwise_distances(x.astype(bool, copy=False), 
                                y.astype(bool, copy=False), 
                                metric='jaccard',n_jobs=n_jobs)
    return 1 - jdist


def funcJaccardSklearn(x, y=None, n_jobs=1):
    x, y  = check_pairwise_arrays(x, y)
    if isspmatrix(x): return 1 - _SparseTanimotoSimilarities(x, y)
    if (x.ndim == 1) and (y.ndim ==1):
        jdist = pairwise_distances(x.astype(bool, copy=False).reshape(1,-1), 
                                y.astype(bool, copy=False).reshape(1,-1), 
                                metric='jaccard',n_jobs=n_jobs)
    else:
        jdist = pairwise_distances(x.astype(bool, copy=False), 
                                y.astype(bool, copy=False), 
                                metric='jaccard',n_jobs=n_jobs)
    return jdist


def ProductTanimotoKernel(x, y=None, len_first=8192, n_jobs=1):
    x, y = check_pairwise_arrays(x, y)
    x1, x2 = x[:,:len_first], x[:,len_first:]
    y1, y2 = y[:,:len_first], y[:,len_first:]
    k1 = funcTanimotoSklearn(x1, y1, n_jobs=n_jobs)
    k2 = funcTanimotoSklearn(x2, y2, n_jobs=n_jobs)
    kproduct = k1*k2
    return kproduct


def AverageTanimotoKernel(x, y=None, len_first=8192, n_jobs=1):
    x, y = check_pairwise_arrays(x, y)
    x1, x2 = x[:,:len_first], x[:,len_first:]
    y1, y2 = y[:,:len_first], y[:,len_first:]
    k1 = funcTanimotoSklearn(x1, y1, n_jobs=n_jobs)
    k2 = funcTanimotoSklearn(x2, y2, n_jobs=n_jobs)
    kave = (k1 + k2) / 2
    return kave


def ProductJaccardKernel(x, y=None, len_first=8192, n_jobs=1):
    x, y = check_pairwise_arrays(x, y)
    x1, x2 = x[:,:len_first], x[:,len_first:]
    y1, y2 = y[:,:len_first], y[:,len_first:]
    k1 = funcTanimotoSklearn(x1, y1, n_jobs=n_jobs)
    k2 = funcTanimotoSklearn(x2, y2, n_jobs=n_jobs)
    kproduct = k1*k2
    return 1 - kproduct


def _fsiminrow(mol_x, mol_y):
    fraggle_similarity = []
    for m_x, m_y in zip(mol_x, mol_y):
        sim, match_ = FraggleSim.GetFraggleSimilarity(m_x, m_y)
        fraggle_similarity.append(sim)
    return np.array(fraggle_similarity)


def FraggleSimilarities(mol_x,mol_y=None,n_jobs=1):
    """
    inputs
    -------
    mol_x : 

    mol_y : 

    Returns
    -------
    D : ndarray of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_samples_Y)
        A similarity matrix D such that D_{i, j} is the similarity between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the similarity between the ith array
        from X and the jth array from Y.
    """
    mol_x_np = np.array(mol_x)
    mol_y_np = np.array(mol_y) if mol_y is not None else mol_x_np.copy()
    n_mol_x  = mol_x_np.shape[0]
    n_mol_y  = mol_y_np.shape[0]
    ret = Parallel(n_jobs=n_jobs,backend='threading')(
        delayed(_fsiminrow)(mol_x_np,mol_y_np[np.repeat(i,n_mol_x)])
        for i in range(n_mol_y)
    )
    return np.stack(ret)

if __name__=='__main__':
    from rdkit import Chem
    sim = FraggleSimilarities([Chem.MolFromSmiles('c1ccccc1')],[Chem.MolFromSmiles('c1ccccc1CNC')])
    