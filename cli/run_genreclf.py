from os.path import join, dirname
import sys
import argparse
sys.path.append(join(dirname(__file__), '..'))

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

from tasks.autotagging import split_data, get_model_instance
from tasks.classification import load_data, instantiate_clf, eval_clf


def setup_argparse():
    # setup argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('genreclf_fn', type=str,
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_argparse()

    # get full factorial design
    configs = json.load(open(args.configure_fn))

    # load the relevant data
    X, y = load_data(args.genreclf_fn, args.feature_fn)

    # run
    spliter = StratifiedShuffleSplit(train_size=0.8)
    i = args.target_row
    conf = configs[i]

    print(conf)
    print('Running {:d}th / {:d} run...'.format(i+1, len(configs)))
    result = []
    # prepare feature according to the design
    x = {k:v for k, v in X.items() if conf[k]}
    for j in range(args.n_rep):
        # split data
        (Xtr, Xvl, Xts), (ytr, yvl, yts) = split_data(x, y, spliter)

        # find best model 
        model = get_model_instance(
            conf['model'], (Xtr, ytr), (Xvl, yvl),
            model_init_f = instantiate_clf,
            eval_f = eval_clf
        )

        # prep test
        X_ = np.concatenate([Xtr, Xvl], axis=0)
        y_ = np.concatenate([ytr, yvl], axis=0)
        test = eval_clf(model, (X_, y_), (Xts, yts))
        print(test)

        # register results
        metrics = {'trial': j, 'score': np.mean(test)}
        conf_copy = conf.copy()
        conf_copy.update(metrics)
        result.append(conf_copy)

    # save
    pd.DataFrame(result).to_csv(args.out_fn)
