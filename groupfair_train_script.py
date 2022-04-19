import numpy as np
import pandas as pd
import copy
import pickle
import itertools
import sklearn
from sklearn.linear_model import LogisticRegression
import os
import time
import argparse
import importlib
import random
import sys

from heterogeneity import create_dataset, split_dataset
from eval_rank import create_dataset as create_dataset_evalrank
from eval_rank import split_dataset as split_dataset_evalrank

sys.path.append('./import_files')
import groupfair_utils as utils

def calc_frontier(method, params_combs, metrics, log=True, trial=-1):
    frontier = []
    all_val_preds = []
    all_test_preds = []
    all_train_preds = []
    for i,params in enumerate(utils.product_dicts(params_combs)):
        time1=time.time()
        learner = method(**params)
        learner.train(tr_X, tr_Xp, tr_y)
        train_time = time.time() - time1
        
        time1=time.time()
        train_preds = learner.predict(tr_X, tr_Xp)
        test_preds = learner.predict(test_X, test_Xp)
        predict_time = time.time()-time1
        time2=time.time()
        val_preds = learner.predict(val_X, val_Xp)

        point_dict = {'method': learner.name, 'params': params}
        metric_dict = {'train_time': train_time, 'predict_time': predict_time}
        for metric_name, metric in exp_metrics:
            metric_dict['train_'+metric_name] = metric(train_preds, tr_X, tr_Xp, tr_y)
            metric_dict[metric_name] = metric(test_preds, test_X, test_Xp, test_y)
        point_dict['metrics'] = metric_dict 
        if log:
            print("=======================================================================================================")
            print(f"{i}th param set for {learner.name} done, train time: {train_time:.5f}, predict_time: {predict_time:.5f}")
            for metric_name, metric in exp_metrics:
                print(f"train_{metric_name}: {metric_dict['train_'+metric_name]:.7f}, {metric_name}: {metric_dict[metric_name]:.7f}")
        frontier.append(point_dict)
        print("Frontier takes: {:.4f}".format((time.time()-other_time)/60.))
        all_val_preds.append(val_preds)
        all_test_preds.append(test_preds)
        all_train_preds.append(train_preds)
    return frontier, all_val_preds, all_test_preds, all_train_preds

def average_dicts(dicts):
    avg_dict = {}
    keys = dicts[0].keys()
    n = len(dicts)
    for key in keys:
        avg_dict[key] = np.mean([dict_[key] for dict_ in dicts])
    return avg_dict

def average_frontiers(frontiers_list):
    avg_frontiers = []
    matched_frontiers = zip(*frontiers_list)
    for matched_frontier_list in matched_frontiers: 
        frontier = []
        matched_points = zip(*matched_frontier_list)
        for matched_point in matched_points:
            expd = matched_point[0] 
            point_dict = {'method': expd['method'], 'params': expd['params']}
            point_dict['metrics'] = average_dicts([pd['metrics'] for pd in matched_point])
            frontier.append(point_dict)
        avg_frontiers.append(frontier)
    return avg_frontiers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expfile', type=str)
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--num_trials', type=int, default=5, help='number of trials')
    parser.add_argument('--dataset', type=int, default=0, help='which of 5 folktable datasets from 2018')
    parser.add_argument('--group', type=int, default=0, help='0 is API, 1 is AI+AN, 2 is other and 3 is biracial (only applicable for other_exp)')
    parser.add_argument('--tier', type=int, default=0, help='0 is args.group is all together, 1 is 2 groups, 2 is all disaggregated')
    parser.add_argument('--other_exp', type=int, default=0, help='0 is regular setting, so other is their own thing, 1 is you try to distribute each of args.group into a different group, and 2 is remove all other, 3 is also regular setting but you just set this to know and it drops rac2p rac3p')
    parser.add_argument('--protatt', type=int, default=0, help='0 is protatt a feature, 1 is no protected attributes, 2 is protatt only as race-sex')
    parser.add_argument('--bwmf', action='store_true', default=False, help='if set, only [Black, white] x [female, male] are kept. This also means that now gender is considered')
    parser.add_argument('--hetero_bw_api', action='store_true', default=False, help='if set, for the hetero api experiment, all non-Black/white/API are discarded')
    parser.add_argument('--bf_train', type=int, default=-1, help='just trained on one group')
    parser.add_argument('--bf_smalln', type=int, default=-1, help='the number of bw, if -1 then just default number')
    parser.add_argument('--fairness_alg', type=int, default=0, help='all of the different methods we have. if -1 then does not train, makes dataset for yang')
    parser.add_argument('--version', type=int, default=0, help='prot att version, 0 is [Black, white] x [male, female], 1 is RAC1P x DIS, 2 is MAR x SEX, 3 is MAR x SEX x DIS')
    parser.add_argument('--evalrank', action='store_true', default=False)
    parser.add_argument('--warnings', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    expfile = importlib.import_module(f"import_files.groupfair.experiments.{args.expfile}")

    if args.seed == -1:
        args.seed = np.random.randint(0, 1000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print("Random seed: {}".format(args.seed))

    if args.evalrank:
        features, label, rac1p_names, states, nums_ordering, atts = create_dataset_evalrank(args)
    else:
        features, label, rac1p_names, remove_names, states, race_feats, nums_ordering = create_dataset(args)

    total_frontiers = []
    all_results = []
    for i in range(args.num_trials):

        if args.evalrank:
            X_train, y_train, X_val, y_val, X_test, y_test, all_indices, nums_ordering, labels_train = split_dataset_evalrank(args, features, label, rac1p_names, states, nums_ordering, atts)
        else:
            X_train, y_train, X_val, y_val, X_test, y_test, all_indices, mid_indices, granular_indices, nums_ordering, labels_train = split_dataset(args, features, label, rac1p_names, remove_names, states, race_feats, nums_ordering)

        the_ys = [y_train, y_val, y_test]
        to_save = []
        for x, X in enumerate([X_train, X_val, X_test]):
            Xp = []
            for ind, index in enumerate(all_indices[x]):
                if args.other_exp == 1:
                    if ind == 6 and args.dataset in [0, 1, 3, 4]:
                        continue
                    if ind == 5 and args.dataset == 2:
                        continue
                this_Xp = np.zeros(len(X))
                this_Xp[np.array(list(index)).astype(int)] = 1
                Xp.append(this_Xp)
            Xp = np.array(Xp).T
            to_save.append([np.array(X), Xp, np.array(the_ys[x]).astype(int)])

        for x in range(3):
            if x == 0:
                tr_X, tr_Xp, tr_y = to_save[x]
            elif x == 1:
                val_X, val_Xp, val_y = to_save[x]
            elif x == 2:
                test_X, test_Xp, test_y = to_save[x]

        frontiers = []
        for method, params_combs, exp_metrics in zip(expfile.methods, expfile.params_list, expfile.metrics_list):
            frontier, probs_val, probs_test, probs_train = calc_frontier(method, params_combs, exp_metrics, trial=i)
            frontiers.append(frontier)
            if args.evalrank:
                trial_results = [y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, nums_ordering]
            else:
                trial_results = [y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, mid_indices, granular_indices, nums_ordering]
            all_results.append(trial_results)


        total_frontiers.append(frontiers)

    average_frontiers = average_frontiers(total_frontiers)


    if not os.path.exists('interpret_results/results'):
        os.makedirs('interpret_results/results')

    if args.bwmf:
        pickle.dump([args, all_results], open('interpret_results/results/alg{0}_d{1}_p{2}_n{3}_bf{4}.pkl'.format(args.expfile, args.dataset, args.protatt, args.bf_smalln, args.bf_train), 'wb'))
    elif args.other_exp > 0:
        pickle.dump([args, all_results], open('interpret_results/results/alg{0}_d{1}_g{2}_o{3}.pkl'.format(args.expfile, args.dataset, args.group, args.other_exp), 'wb'))
    elif args.evalrank:
        pickle.dump([args, all_results], open('interpret_results/results/evalrank_alg{0}_d{1}_v{2}.pkl'.format(args.expfile, args.dataset, args.version), 'wb'))
    else:
        pickle.dump([args, all_results], open('interpret_results/results/alg{0}_d{1}_g{2}_t{3}_bw{4}.pkl'.format(args.expfile, args.dataset, args.group, args.tier, '1' if args.hetero_bw_api else '0'), 'wb'))


