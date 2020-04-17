from itertools import combinations

import numpy as np
import numba as nb
from scipy import sparse as sp

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

import h5py

from .files import mxm2msd as mxm2msd_fn


def prepare_feature(feature_fn):
    """"""
    # getting mxm->msd map
    mxm2msd = dict(
        [line.strip('\n').split(',') for line in open(mxm2msd_fn())]
    )

    # load the feature data and concatenate
    with h5py.File(feature_fn, 'r') as hf:

        features = {}
        bounds = [0]
        feature_sets = [
            k.split('_cols')[0]
            for k in hf['features'].keys()
            if 'cols' in k
        ]

        for feat in feature_sets:
            # fetch features per set
            features[feat] = hf['features'][feat][:]

        track2id = {
            mxm2msd[t]:i for i, t
            in enumerate(hf['features']['ids'][:])
        }

    return features, track2id


def preproc_feat(X, split_idx):
    """"""
    # prepare container
    Xout = {split:{} for split in split_idx.keys()}
    # for some pre-processors
    pca = PCA(whiten=True)
    sclr = StandardScaler()
    feat_names = list(X.keys())
    for name in feat_names:
        if name == 'topic': proc = pca
        else:               proc = sclr

        for split, ix in split_idx.items():
            # retrieve slice of the data
            Xout[split][name] = X[name][ix]

            if name == 'audio':
                # get the non-zero / zero entries
                xsum = Xout[split][name].sum(1)
                nnz_idx = np.where(xsum > 0)[0]
                zro_idx = np.where(xsum == 0)[0]

                # if it's training set, fit the processor
                if split == 'train':
                    proc.fit(Xout[split][name][nnz_idx])

                # process non-zero rows
                Xout[split][name][nnz_idx] = proc.transform(
                    Xout[split][name][nnz_idx]
                )
                # assign random vectors to the zero rows
                Xout[split][name][zro_idx] = np.random.randn(
                    len(zro_idx), X[name].shape[1]
                )
            else:
                # normal case
                if split == 'train': proc.fit(Xout[split][name])
                Xout[split][name] = proc.transform(Xout[split][name])

    Xout = {
        split:np.concatenate([x[name] for name in feat_names], axis=1)
        for split, x in Xout.items()
    }
    return Xout, feat_names


def split_data(X, y, spliter=None, train_ratio=0.8):
    """"""
    if spliter is None:
        spliter = ShuffleSplit(train_size=train_ratio)

    tr_ix, ts_ix = next(spliter.split(y, y))
    tr_ix_, vl_ix_ = next(spliter.split(y[tr_ix], y[tr_ix]))
    split_idx = {
        'train':tr_ix[tr_ix_],
        'valid':tr_ix[vl_ix_],
        'test':ts_ix
    }

    # preprocess the data
    x, feat_cols = preproc_feat(X, split_idx)
    Xtr, Xvl, Xts = x['train'], x['valid'], x['test']
    ytr, yvl, yts = (
        y[split_idx['train']],
        y[split_idx['valid']],
        y[ts_ix]
    )

    return (Xtr, Xvl, Xts), (ytr, yvl, yts)


def load_csr_data(h5py_fn, row='users', col='items'):
    """ Load recsys data stored in hdf format

    Inputs:
        fn (str): filename for the data

    Returns:
        scipy.sparse.csr_matrix: user-item matrix
        numpy.ndarray: user list
        numpy.ndarray: item list
    """
    import h5py
    with h5py.File(h5py_fn, 'r') as hf:
        data = (hf['data'][:], hf['indices'][:], hf['indptr'][:])
        X = sp.csr_matrix(data)
        rows = hf[row][:]
        cols = hf[col][:]
    return X, rows, cols


def get_all_comb(cases, include_null=False):
    combs = [
        combinations(cases, j)
        for j in range(1, len(cases) + 1)
    ]
    if include_null:
        combs.append(None)
    return combs


def mat2array(matrix, flatten=False):
    array = np.array(matrix)
    if flatten:
        return array.ravel()
    else:
        return array


@nb.njit
def vecmat(vec, mat):
    # assuming len(vec) == mat.shape[0]
    out = np.zeros((mat.shape[1],), dtype=mat.dtype)
    for i in range(len(vec)):
        for j in range(mat.shape[1]):
            out[j] += vec[i] * mat[i, j]
    return out


def slice_row_sparse(csr, i):
    slc = slice(csr.indptr[i], csr.indptr[i+1])
    return csr.indices[slc], csr.data[slc]


def argpart_sort(s, k, ascending=True):
    if ascending: p = s
    else:         p = -s
    idx = np.argpartition(p, kth=k)[:k]
    return idx[np.argsort(p[idx])]


def argpart_sort_2d(s, k, ascending=True):
    if ascending: p = s
    else:         p = -s
    n = p.shape[0]
    rng = np.arange(n)[:, None]
    idx = np.argpartition(p, kth=k, axis=1)[:, :k]
    inner_idx = np.argsort(p[rng, idx], axis=1)
    rec = idx[rng, inner_idx]
    return rec


def densify(ui_csr, users, items, item_feat=None, thresh=5, user_sample=0.3):
    """ Densify the User-Item interactio matrix
    """
    def _filt_entity(csr, entities, thresh):
        filt_targs = np.where(np.ediff1d(csr.indptr) >= thresh)[0]
        return csr[filt_targs], entities[filt_targs], filt_targs

    n_users, n_items = ui_csr.shape
    users = np.asarray(users)
    items = np.asarray(items)

    if user_sample > 0:
        assert user_sample < 1
        p = user_sample
        uid = np.random.choice(n_users, int(n_users * p), False)
        ui_csr = ui_csr[uid]
        users = users[uid]

    diff = 1
    while diff > 0:
        prev_nnz = ui_csr.nnz
        iu_csr, items, filt_idx = _filt_entity(ui_csr.T.tocsr(), items, thresh)
        if item_feat is not None:
            item_feat = item_feat[filt_idx]
        ui_csr, users, filt_idx = _filt_entity(iu_csr.T.tocsr(), users, thresh)
        diff = prev_nnz - ui_csr.nnz
    return ui_csr, users, items, item_feat
