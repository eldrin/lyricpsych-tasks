from os.path import join, dirname
import sys
import argparse
sys.path.append(join(dirname(__file__), '..'))

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import ShuffleSplit

from tasks.recsys import (load_data,
                          eval_model,
                          get_model_instance,
                          instantiate_model)
from tasks.utils import split_data, densify


def setup_argparse():
    # setup argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('recsys_fn', type=str,
                        help='filename of the auto-tagging label data')
    parser.add_argument('feature_fn', type=str,
                        help='filename to the lyrics feature file')
    parser.add_argument('configure_fn', type=str,
                        help='filename to the design file (.json)')
    parser.add_argument('target_row', type=int,
                        help='target row of configuration table to run')
    parser.add_argument('--out-fn', type=str, default='test.csv',
                        help='filename for the test output')
    parser.add_argument('--n-rep', type=int, default=5,
                        help='number of testing users')
    parser.add_argument('--fm-batch-sz', type=int, default=256,
                        help='batch size for fitting factorization machine')
    parser.add_argument('--fm-gpu', dest='fm_gpu', action='store_true')
    parser.set_defaults(fm_gpu=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_argparse()

    # get full factorial design
    configs = json.load(open(args.configure_fn))

    # load the relevant data
    print('Loading data...')
    X, Y, users, items = load_data(args.recsys_fn, args.feature_fn)
    items2ix = {tid:i for i, tid in enumerate(items)}

    print('Densifying the data...')
    X, users, items_new, _ = densify(X, users, items)
    item_filt_ix = [items2ix[tid] for tid in items_new]
    Y = {k:v[item_filt_ix] for k, v in Y.items()}

    # run
    spliter = ShuffleSplit(train_size=0.8)
    i = args.target_row
    conf = configs[i]

    print(conf)
    print('Running {:d}th / {:d} run...'.format(i+1, len(configs)))
    result = []
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
                use_gpu=args.fm_gpu
            )
        else:
            model = instantiate_model(conf['model'],
                                      use_gpu=args.fm_gpu,
                                      **params)

        # prep test
        X_ = sp.hstack([Xtr, Xvl]).tocsr()
        Y_ = np.vstack([Ytr, Yvl])
        test = eval_model(model, (X_, Y_), (Xts, Yts))
        print(np.mean(test))

        # register results
        metrics = {'trial': j, 'score': np.mean(test)}
        conf_copy = conf.copy()
        conf_copy.update(metrics)
        result.append(conf_copy)

    # save
    pd.DataFrame(result).to_csv(args.out_fn)
