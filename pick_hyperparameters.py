import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.stats import kendalltau
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
import argparse
from sklearn.linear_model import LogisticRegression
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSEmployment, ACSRace
from folktables import ACSIncomeDem, ACSPublicCoverageDem, ACSTravelTimeDem, ACSMobilityDem, ACSEmploymentFilteredDem # 5 from paper, with demographic of races
import time
import sys
from scipy.stats import ttest_ind
from scipy.stats.mstats import gmean

sys.path.append('./import_files')
from fairness_algs import *
from all_algs import *

parser = argparse.ArgumentParser(description='intersectionality')
parser.add_argument('--experiment', type=int, default=0, help='0 matches interpret_hetero.py and is api, 1 matches interpret_other.py, 2 matches interpret_evalrank.py')
parser.add_argument('--test', action='store_true', default=False, help='if set true, then test instead of val')
args = parser.parse_args()

algs = {'Baseline': [0, 1, 2, 3, 4, 5], 'RWT': [[6, 7, 8, 9], [10, 11, 12, 13]], 'RDC': [14, 15, 16, 17, 18, 19], 'LOS': [20, 21, 22, 23, 24, 25, 26, 27, 28], 'GRP': ['eo_exp_plugin'], 'GRY': [29, 30, 31, 32, 33, 34]}
results = {}
stds = {}
start = time.time()


num_variations = 0
if args.experiment in [0, 1]:
    num_variations = 3
elif args.experiment in [2]:
    num_variations = 4
else:
    assert NotImplementedError


result_folder = 'interpret_results/results'

for alg in algs.keys():
    print("Algorithm {}".format(alg))
    for d in range(5):
        for t in range(num_variations):
            if args.experiment == 0:
                key = '{0}-d{1}-t{2}'.format(alg, d, t)
            elif args.experiment == 1:
                o = t+1
                key = '{0}-d{1}-o{2}'.format(alg, d, o)
            elif args.experiment == 2:
                key = '{0}-d{1}-v{2}'.format(alg, d, t)
            results[key] = []
            stds[key] = []
            these_algs = algs[alg]
            if alg in ['RWT']:
                if d in [1, 2, 4]:
                    these_algs = these_algs[0]
                else:
                    these_algs = these_algs[1]
            if 'GRP' in alg:
                if alg == 'GRP_ERM':
                    if (d not in [2]) or (args.experiment == 2):
                        continue
                if args.experiment == 1 and o == 2:
                    continue
                these_algs = algs[alg]*5
            for ha, hyper_alg in enumerate(these_algs):
                if args.experiment == 0:
                    file_name = '{5}/alg{0}_d{1}_g{2}_t{3}_bw{4}.pkl'.format(hyper_alg, d, 0, t, '1', result_folder)
                elif args.experiment == 1:
                    file_name = '{4}/alg{0}_d{1}_g{2}_o{3}.pkl'.format(hyper_alg, d, 2, o, result_folder)
                elif args.experiment == 2:
                    file_name = '{3}/evalrank_alg{0}_d{1}_v{2}.pkl'.format(hyper_alg, d, t, result_folder)

                try:
                    these_args, all_results = pickle.load(open(file_name, 'rb'))
                except:
                    results[key].append(-1)
                    stds[key].append(-1)
                    print("Missing file: {}".format(file_name))
                    continue
                
                this_results = []
                for i in range(len(all_results)):
                    if args.experiment in [0, 1]:
                        y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, mid_indices, granular_indices, nums_ordering = all_results[i]
                    elif args.experiment in [2]:
                        y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, nums_ordering = all_results[i]
                    if 'GRP' in alg:
                        probs_val, probs_test = probs_val[ha], probs_test[ha]
                    if args.experiment == 0:
                        indices_to_consider = granular_indices
                    elif args.experiment == 1:
                        indices_to_consider = mid_indices
                    elif args.experiment == 2:
                        indices_to_consider = all_indices

                    soft_acc = (y_val*probs_val + (1.-y_val)*(1.-probs_val)).mean()
                    tprs = []
                    y_1 = np.where(y_val==1)[0]
                    for g in range(len(indices_to_consider[1])):
                        these_indices = np.array(list(set(y_1)&set(indices_to_consider[1][g])))
                        if len(these_indices) > 0:
                            tprs.append(np.mean(probs_val[these_indices]))
                        else:
                            print("No positives in group {0}")
                    tpr_diff = np.amax(tprs)-np.amin(tprs)
                    combo = gmean([soft_acc, 1.-tpr_diff])
                    this_results.append(combo)

                results[key].append(np.mean(this_results))
                stds[key].append(np.std(this_results))
            if 'GRP' in alg:
                try:
                    best_hp = np.argmax(results[key])
                    for i in range(len(all_results)):
                        for probs_ind in [3, 4, 5]:
                            all_results[i][probs_ind] = all_results[i][probs_ind][best_hp]
                            probs = all_results[i][probs_ind]
                    file_name = file_name.replace(result_folder, 'interpret_results/results')
                    pickle.dump([these_args, all_results], open(file_name, 'wb'))
                    results[key] = [0]
                except:
                    print("Couldn't save new format for {}".format(alg))
            if np.amax(results[key]) == -1:
                print("None worked for Alg {0}, setting {1}".format(alg, key))
print("Took {} min".format((time.time()-start)/60.))
pickle.dump([results, algs], open('interpret_results/hyperparameters/v{}.pkl'.format(args.experiment), 'wb'))
