import os
import numpy as np
import torch
from scipy import sparse as sp
from implicit.als import AlternatingLeastSquares

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# from aarms.models.als_feat import ALSFeat
from aarms.models.als import ALS
from factormach.models.uifm import UserItemFM as FactorizationMachine
from factormach.metric import ndcg

from tqdm import tqdm

from .utils import (argpart_sort,
                    argpart_sort_2d,
                    load_csr_data,
                    prepare_feature,
                    slice_row_sparse,
                    densify, mat2array, vecmat)
from .autotagging import full_factorial_design


class Recsys:
    def _recommend(self, user, user_item, item_feat, n):
        """ Internal function that actually serves the rec
        """
        raise NotImplementedError('[ERROR] this should be implemented!')

    def _update(self, new_item_feat):
        """ Update the model for incoming new items

        The updated item idx is assigned after the item indices for trianing items
        (ie. # of training items (0 .. 20) and # of new items 10 then it's (21 .. 30))
        """
        raise NotImplementedError('[ERROR] this should be implemented!')

    def fit(self, user_item, item_feat, *args, **kwargs):
        """ Fit the recommender model if needed
        """
        raise NotImplementedError('[ERROR] this should be implemented!')

    def recommend(self, user, user_item, n=100):
        """ Helper function that does the pre/post processing
        """
        slc = slice(user_item.indptr[user], user_item.indptr[user+1])
        gt = user_item.indices[slc]
        item = self._recommend(user, user_item, n, gt)
        return item


class ItemKNN(Recsys):
    def __init__(self, k):
        """
        """
        super().__init__()
        self.k = k

    def fit(self, user_item, item_feat, verbose=False):
        self.item_profiles_ = item_feat
        self.item_profiles_ /= np.linalg.norm(
            self.item_profiles_, axis=1
        )[:, None]

        # (maybe) build a item-item approximated graph


    def _update(self, new_item_feat):
        self.item_profiles_ = np.vstack([self.item_profiles_, new_item_feat])

    def _recommend(self, user, user_item, n, gt=None):
        """
        """
        items, _ = slice_row_sparse(user_item, user)
        d = self.item_profiles_[items] @ self.item_profiles_.T

        # get the K nearest item neighbors
        neighbors = argpart_sort_2d(d, self.k)

        # score (local-freq)
        s = np.zeros((user_item.shape[1],))
        s_ = np.bincount(neighbors.ravel())
        s[:len(s_)] = s_

        if gt is not None:
            s[gt] = -np.inf

        recs = argpart_sort(-s, n)
        return recs


class ItemNeighbor(Recsys):
    def __init__(self, alpha=1, eps=1e-3):
        """
        """
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    @staticmethod
    def _normalize(item_feat):
        return item_feat / np.linalg.norm(item_feat, axis=1)[:, None]

    def fit(self, user_item, item_feat, verbose=False):

        n_users, n_items = user_item.shape
        self.item_profiles_ = self._normalize(item_feat)

        # confidence matrix (minus 1)
        # C = 1 + alpha * R
        # Z = C.sum(1)
        # Z_inv = Z**-1
        CmI = user_item.copy()
        # CmI.data = CmI.data * self.alpha
        CmI.data = np.log(CmI.data / self.eps + 1)

        user_intensity = mat2array(CmI.sum(1), flatten=True).astype(float)

        Z = user_intensity + n_items
        Z_inv = sp.diags(Z**-1)

        self.user_profiles_ = CmI @ self.item_profiles_
        self.user_profiles_ = self.user_profiles_ + self.item_profiles_.sum(0)[None]
        self.user_profiles_ = Z_inv @ self.user_profiles_

    def _update(self, new_item_feat):
        new_item_feat_ = self._normalize(new_item_feat)
        self.item_profiles_ = np.vstack([self.item_profiles_, new_item_feat_])

    def _recommend(self, user, user_item, n, gt=None):
        """
        """
        d = self.user_profiles_[user] @ self.item_profiles_.T
        # d = vecmat(self.user_profiles_[user], self.item_profiles_.T)

        if gt is not None:
            d[gt] = np.inf

        recs = argpart_sort(d, n)
        return recs


class WRMFFeat(Recsys):
    def __init__(self, k, init=0.001, lmbda=1, l2=0.0001, n_iters=15, alpha=5):
        super().__init__()
        self.k = k
        self.init = init
        self.l2 = l2
        self.n_ites = n_iters
        self.alpha = alpha
        self.lmbda = lmbda
        self._model = ALS(k, init, l2, n_iters, alpha, dtype='float64')

    def fit(self, user_item, item_feat, verbose=False):
        self._model.fit(user_item,
                        item_dense_feature=item_feat,
                        lmbda_item_dense_feature=self.lmbda,
                        verbose=verbose)

        # we track down the item factors specially since it can be updated
        self.item_factors_ = self._model.embeddings_['item']

    def _update(self, new_item_feat):
        v = new_item_feat @ self._model.embeddings_['item_dense_feature']
        self.item_factors_ = np.vstack([self.item_factors_, v])

    def _recommend(self, user, user_item, n, gt=None):
        s = self._model.embeddings_['user'][user] @ self.item_factors_.T
        if gt is not None:
            s[gt] = -np.inf

        recs = argpart_sort(-s, n)
        return recs


class FM(Recsys):
    def __init__(self, k, init=0.01, learn_rate=0.001, l2=1e-6, n_iters=50,
                 use_gpu=False, loss='bce', batch_sz=200, n_jobs=2):
        """"""
        self.k = k
        self.init = init
        self.learn_rate = learn_rate
        self.l2 = l2
        self.n_iters = n_iters
        self.batch_sz = batch_sz
        self.n_jobs = n_jobs
        self._model = FactorizationMachine(
            k, init, n_iters, learn_rate, l2, loss=loss, use_gpu=use_gpu,
            no_item_factor=True
        )

    def fit(self, user_item, item_feat, verbose=False):
        """"""
        self._model.fit(user_item, None, item_feat, verbose=verbose,
                        batch_sz=self.batch_sz, n_jobs=self.n_jobs)

        # post-process the model
        self._model._update_z(
            {'item': torch.Tensor(item_feat).to(self._model.device)}
        )
        self._model.eval()

        # setup the internal state
        self.zi, self.zi2 = self._model.zi, self._model.zi2
        self.zu, self.zu2 = self._model.zu, self._model.zu2

    def _update(self, new_item_feat):
        # pre-compute the z vectors
        new_feat = torch.Tensor(new_item_feat).to(self._model.device)
        zfi_vl = new_feat @ self._model.embeddings_['feat_item'].weight
        zfi2_vl = new_feat**2 @ self._model.embeddings_['feat_item'].weight**2
        self.zi = torch.cat([self.zi, zfi_vl], dim=0)[None]
        self.zi2 = torch.cat([self.zi2, zfi2_vl], dim=0)[None]
        self.zu = self.zu[:, None]
        self.zu2 = self.zu2[:, None]

    def _recommend(self, user, user_item, n, gt=None):
        # infer
        w = self.zi[..., -1] + self.zu[user][..., -1]
        v = (self.zi[..., :-1] + self.zu[user][..., :-1])**2
        v -= (self.zi2[..., :-1] + self.zu2[user][..., :-1])
        v = v.sum(-1) * .5
        s = (self._model.w0 + w + v).cpu().data.numpy()[0]
        if gt is not None:
            s[gt] = -np.inf
        recs = argpart_sort(-s, n)
        return recs


class WRMF(Recsys):
    def __init__(self, k, reg=1e-4, n_iters=15):
        """"""
        super().__init__()
        self.k = k
        self.reg = reg
        self.n_iters = n_iters
        self.als = AlternatingLeastSquares(
            k, regularization=reg, iterations=n_iters
        )

    def _recommend(self, user, user_item, n, gt=None):
        return np.array([
            itemid for itemid, score
            in self.als.recommend(user, user_item, n)
        ])


class Random(Recsys):
    def __init__(self):
        super().__init__()

    def _recommend(self, user, user_item, n, gt):
        item = np.random.permutation(user_item.shape[1])
        return item[~np.in1d(item, gt)][:n]


class MostPopular(Recsys):
    def __init__(self, user_item):
        super().__init__()
        self.user_item = user_item
        self.mp = np.argsort(-user_item.sum(0).A.flatten())

    def _recommend(self, user, user_item, n, gt):
        return self.mp[~np.in1d(self.mp, gt)][:n]


def eval_model(model, train_data, test_data,
               n_test_users=2000, topn=100, verbose=False):
    """"""
    # parse data
    Xtr, Ytr = train_data
    Xvl, Yvl = test_data

    # fit the model
    model.fit(Xtr, Ytr, verbose=True)
    model._update(Yvl)

    # prepare the data
    X_test = sp.hstack([Xtr, Xvl]).tocsr()
    X_train = sp.hstack([Xtr, sp.csr_matrix(Xvl.shape)]).tocsr()

    test_scores = []
    if n_test_users <= 0:
        rng = np.arange(Xvl.shape[0])
    else:
        rng = np.random.choice(Xvl.shape[0], n_test_users, False)

    train_bound = Xtr.shape[1]
    with tqdm(total=len(rng), ncols=80, disable=not verbose) as p:
        for u in rng:
            pos, _ = slice_row_sparse(X_test, u)
            train, true = pos[pos < train_bound], pos[pos >= train_bound]
            if len(true) == 0:
                continue
            pred = model.recommend(u, X_train, topn)
            test_scores.append(ndcg(true, pred, k=topn))
            p.update(1)
    return test_scores


def instantiate_model(model_class, k=32, lmbda=1, l2=1e-6, n_iters=50,
                      alpha=1, eps=1e-1, learn_rate=1e-3, batch_sz=256,
                      use_gpu=False):
    """"""
    k = int(k)  # should be cased to integer

    # instantiate a model
    if model_class == 'ItemNeighbor':
        model = ItemNeighbor(alpha=1e+6, eps=1e+6)

    elif model_class == 'WRMFFeat':
        model = WRMFFeat(k, lmbda=lmbda, alpha=alpha, l2=l2)

    elif model_class == 'FM':
        model = FM(k, l2=l2, learn_rate=learn_rate,
                   batch_sz=batch_sz, n_iters=n_iters,
                   use_gpu=use_gpu)

    elif model_class == 'MostPopular':
        model = MostPopular(Xtr)

    elif model_class == 'Random':
        model = Random()

    return model


def get_model_instance(model_class, train_data, valid_data,
                       n_test_users=5000, topn=100, n_opt_calls=50,
                       rnd_state=0, use_gpu=False, batch_sz=256):
    """"""
    # hyper param search range
    search_spaces = {
        'WRMFFeat': [
            Real(1e-7, 1e+1, "log-uniform", name='alpha'),
            Real(1e+4, 1e+10, "log-uniform", name='lmbda'),
            Real(1e-4, 1, "log-uniform", name='l2')
        ]
    }
    Xtr, Ytr = train_data
    Xvl, Yvl = valid_data

    if model_class in search_spaces:
        # setup objective func evaluated by optimizer
        @use_named_args(search_spaces[model_class])
        def _objective(**params):
            model = instantiate_model(model_class,
                                      use_gpu=use_gpu,
                                      batch_sz=batch_sz,
                                      **params)

            # evaluate the model
            scores = eval_model(model, (Xtr, Ytr), (Xvl, Yvl),
                                n_test_users, topn)
            return -np.mean(scores)

        # search best model with gaussian process based
        res_gp = gp_minimize(
            _objective, search_spaces[model_class],
            n_calls=n_opt_calls, random_state=rnd_state,
            verbose=True
        )
        best_param = {
            param.name: val for param, val
            in zip(search_spaces[model_class], res_gp['x'])
        }
        print(res_gp.fun, best_param)
    else:
        best_param = {}

    best_model = instantiate_model(
        model_class,
        use_gpu=use_gpu,
        batch_sz=batch_sz,
        **best_param
    )
    return best_model, best_param


def load_data(recsys_data_fn, text_feat_data_fn):
    """"""
    X, users, items = load_csr_data(recsys_data_fn)
    item_hash_rec_data = {tid:j for j, tid in enumerate(items)}
    feature, track2id = prepare_feature(text_feat_data_fn)

    mutual_tid = [tid for tid in items if tid in track2id]
    ix = [track2id[i] for i in mutual_tid]
    Y = {k:feat[ix] for k, feat in feature.items()}

    mut_tid_set = set(mutual_tid)
    targets = [item_hash_rec_data[tid] for tid in mutual_tid]
    X2 = X[:, targets]
    return X2, Y, users, mutual_tid


def setup_argparser():
    # setup argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('recsys_fn', type=str,
                        help='filename of the recsys interaction data')
    parser.add_argument('feature_fn', type=str,
                        help='filename to the lyrics feature file')
    parser.add_argument('--out-fn', type=str, default='test.csv',
                        help='filename for the test output')
    parser.add_argument('--topn', type=int, default=100,
                        help='truncate threshold for evaulation')
    parser.add_argument('--n-test-users', type=int, default=2000,
                        help='number of testing users')
    parser.add_argument('--n-rep', type=int, default=5,
                        help='number of testing users')
    parser.add_argument('--fm-batch-sz', type=int, default=256,
                        help='batch size for fitting factorization machine')
    parser.add_argument('--fm-gpu', dest='fm_gpu', action='store_true')
    parser.set_defaults(fm_gpu=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # build argparser
    args = setup_argparser()

    # load run specific packages
    from sklearn.model_selection import ShuffleSplit

    # get full factorial design
    configs = full_factorial_design(models=['ItemNeighbor', 'WRMFFeat', 'FM'])

    # load the relevant data
    print('Loading data...')
    X, Y, users, items = load_data(args.recsys_fn, args.feature_fn)
    items2ix = {tid:i for i, tid in enumerate(items)}

    print('Densifying the data...')
    X, users, items_new, _ = densify(X, users, items)
    item_filt_ix = [items2ix[tid] for tid in items_new]
    Y = {k:v[item_filt_ix] for k, v in Y.items()}

    # run
    result = []
    spliter = ShuffleSplit(train_size=0.8)
    for i, conf in enumerate(configs):
        print(conf)
        print('Running {:d}th / {:d} run...'.format(i+1, len(configs)))

        # prepare feature according to the design
        Y = {k:v for k, v in Y.items() if conf[k]}
        for j in range(args.n_rep):
            # split data
            print('Splitting the data...')
            (Ytr, Yvl, Yts), splitted_interaction = split_data(Y, X.T.tocsr())
            (Xtr, Xvl, Xts) = tuple([x.T.tocsr() for x in splitted_interaction])

            if j == 0:
                # find best model
                model, params = get_model_instance(
                    conf['model'], (Xtr, Ytr), (Xvl, Yvl),
                    use_gpu=args.fm_gpu, batch_sz=args.fm_batch_sz
                )
            else:
                model = instantiate_model(
                    conf['model'],
                    use_gpu=args.fm_gpu,
                    batch_sz=args.fm_batch_sz,
                    **params
                )

            # prep test
            X_ = sp.hstack([Xtr, Xvl]).tocsr()
            Y_ = np.vstack([Ytr, Yvl])
            test = eval_model(
                model, (X_, Y_), (Xts, Yts), n_test_users=10000
            )
            print(np.mean(test))

            # register results
            metrics = {'trial': j, 'score': np.mean(test)}
            conf_copy = conf.copy()
            conf_copy.update(metrics)
            result.append(conf_copy)

    # save
    pd.DataFrame(result).to_csv(args.out_fn)
