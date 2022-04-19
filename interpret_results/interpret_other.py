import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import pickle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
import copy
import pandas as pd
import numpy as np
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
from brokenaxes import brokenaxes

sys.path.append('./import_files')
from fairness_algs import *
from all_algs import *

def soft_acc(y, probs):
    return (y*probs + (1.-y)*(1.-probs)).mean()

parser = argparse.ArgumentParser(description='intersectionality')
parser.add_argument('--dataset', type=int, default=0, help='which of 5 folktable datasets from 2018')
parser.add_argument('--group', type=int, default=2, help='0 is API, 1 is AI+AN, 2 is other')
parser.add_argument('--test', action='store_true', default=False, help='if set true, then test instead of val')
args = parser.parse_args()

to_view = 0
constant = -1
min_val = 1

results, algs = pickle.load(open('interpret_results/hyperparameters/v1.pkl', 'rb'))

all_algs = ['Baseline', 'RWT', 'RDC', 'LOS', 'GRP', 'GRY']
plt.figure(figsize=(6, 3))

for ind, label in enumerate(all_algs):
    metric_names = ['acc', 'auc', 'tpr']
    print("---Alg {}---".format(label))
    compare = {}
    for other_exp in [1, 2, 3]:

        if other_exp == 2 and label in ['GRY', 'GRP']:
            continue

        #### picking from hyperparameters
        key = '{0}-d{1}-o{2}'.format(label, args.dataset, other_exp)
        these_algs = algs[label]
        if label in ['RWT']:
            if args.dataset in [1, 2, 4]:
                these_algs = these_algs[0]
            else:
                these_algs = these_algs[1]
        method = these_algs[np.argmax(results[key])]
        #### picking from hyperparameters
        metric_values = [[] for _ in range(len(metric_names))]


        these_args, all_results = pickle.load(open('interpret_results/results/alg{0}_d{1}_g{2}_o{3}.pkl'.format(method, args.dataset, args.group, other_exp), 'rb'))
        for i in range(len(all_results)):
            y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, mid_indices, granular_indices, nums_ordering = all_results[i]
            if args.group == 2:
                other_group_idx = list(nums_ordering[0]).index('Other')
            else:
                raise NotImplementedError

            if args.test:
                y_this, probs_this = y_test, probs_test
                index_this = 2
            else:
                y_this, probs_this = y_val, probs_val
                index_this = 1

            tprs = []
            y_1 = np.where(y_this==1)[0]
            for g in range(len(mid_indices[index_this])):
                these_indices = np.array(list(set(y_1)&set(mid_indices[index_this][g])))
                if len(these_indices) > 0:
                    tprs.append(np.mean(probs_this[these_indices]))
                else:
                    if g != other_group_idx:
                        print("No positives in group {0}".format(g))
            tpr_diff = np.amax(tprs)-np.amin(tprs)

            other_indices = mid_indices[index_this][other_group_idx] 
            metric_values[0].append(soft_acc(y_this[other_indices], probs_this[other_indices]))
            other_auc = roc_auc_score(y_this[mid_indices[index_this][other_group_idx]], probs_this[mid_indices[index_this][other_group_idx]])
            metric_values[1].append(other_auc)
            
            metric_values[2].append(tprs[other_group_idx])

        min_val = min(min_val, np.mean(metric_values[to_view]))
        if other_exp == 1:
            if constant > 0:
                bax.bar([ind+2*.3], [np.mean(metric_values[to_view])], width=.3, color='C{}'.format(ind), yerr=np.std(metric_values[to_view]), edgecolor='k', hatch='/') 
            else:
                plt.bar([ind+2*.3], [np.mean(metric_values[to_view])], width=.3, color='C{}'.format(ind), yerr=np.std(metric_values[to_view]), edgecolor='k', hatch='/')
        elif other_exp == 2:
            if label in ['GRY', 'GRP']:
                continue
            if constant > 0:
                bax.bar([ind+3*.3], [np.mean(metric_values[to_view])], width=.3, color='C{}'.format(ind), yerr=np.std(metric_values[to_view]), hatch='o', edgecolor='k')
            else:
                plt.bar([ind+3*.3], [np.mean(metric_values[to_view])], width=.3, color='C{}'.format(ind), yerr=np.std(metric_values[to_view]), hatch='o', edgecolor='k')
        elif other_exp == 3:
            if constant > 0:
                bax.bar([ind+1*.3], [np.mean(metric_values[to_view])], width=.3, color='C{}'.format(ind), yerr=np.std(metric_values[to_view]), edgecolor='k')
            else:
                plt.bar([ind+1*.3], [np.mean(metric_values[to_view])], width=.3, color='C{}'.format(ind), yerr=np.std(metric_values[to_view]), edgecolor='k')

        compare[other_exp] = metric_values[to_view]
    for key1 in compare.keys():
        for key2 in compare.keys():
            if key1 < key2:
                t, p = ttest_ind(compare[key1], compare[key2])
                print("Alg {0} with other exp {1} and {2}, p: {3:.4f}, t: {4:.4f}".format(label, key1, key2, p, t))


plt.bar([0], [0], width=0, edgecolor='k', color='w', label='Separate')
plt.bar([0], [0], width=0, edgecolor='k', color='w', label='Redistribute', hatch='/')
plt.bar([0], [0], width=0, edgecolor='k', color='w', label='Ignore', hatch='o')

fontsize = 11

plt.legend(prop={'size': 8.5})
plt.xticks(np.arange(len(algs))+.6, all_algs, fontsize=fontsize)
plt.xlabel('Algorithm', fontsize=fontsize)
plt.ylim(bottom=min_val-.01)
if args.dataset == 2:
    plt.ylim(bottom=min_val-.01, top=.83)
if to_view == 0:
    plt.ylabel('Other Acc'.format(constant), fontsize=fontsize)
elif to_view == 1:
    plt.ylabel('Other AUC'.format(constant), fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()
plt.savefig('interpret_results/images/other_d{}.png'.format(args.dataset), dpi=300)
plt.close()
print("-----")

