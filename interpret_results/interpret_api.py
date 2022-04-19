import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from scipy.stats import ttest_ind

sys.path.append('./import_files')
from fairness_algs import *
from all_algs import *

parser = argparse.ArgumentParser(description='intersectionality')
parser.add_argument('--dataset', type=int, default=0, help='which of 5 folktable datasets from 2018')
parser.add_argument('--group', type=int, default=0, help='0 is API, 1 is AI+AN')
parser.add_argument('--method', type=int, default=0, help='defined in ../clean_code/vis_curves.py')
parser.add_argument('--hetero_bw_api', action='store_false', default=True, help='if set, for the hetero api experiment, all non-Black/white/API are disarded')
parser.add_argument('--test', action='store_true', default=False, help='if set true, then test instead of val')
args = parser.parse_args()

cmap = plt.get_cmap('Set3')
cmap.colors = list(cmap.colors)
cmap.colors[1] = 'k'
fontsize = 15

# mid indices obsolete, but whatever
names = []
if args.group == 0:
    for t in range(3):
        these_args, all_results = pickle.load(open('interpret_results/results/alg{0}_d{1}_g{2}_t{3}_bw{4}.pkl'.format('3', args.dataset, args.group, t, '1' if args.hetero_bw_api else '0'), 'rb'))
        y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, mid_indices, granular_indices, nums_ordering = all_results[0]
        names.append(nums_ordering[0])
api_for_granular = []
for n, name in enumerate(names[2]):
    if 'Asian' in name or 'Pacific' in name:
        api_for_granular.append(n)
api_for_mid = []
for n, name in enumerate(names[1]):
    if 'Asian' in name or 'Pacific' in name:
        api_for_mid.append(n)

to_view = 3

results, algs = pickle.load(open('interpret_results/hyperparameters/v0.pkl', 'rb'))

plt.figure(figsize=(6, 3.2))
all_algs = ['Baseline', 'RWT', 'RDC', 'LOS', 'GRP', 'GRY']

for ind, label in enumerate(all_algs):
    print("---Alg {}---".format(label))
    compare = {}
    for t in range(3):

        #### picking from hyperparameters
        key = '{0}-d{1}-t{2}'.format(label, args.dataset, t)
        these_algs = algs[label]
        if label in ['RWT']:
            if args.dataset in [1, 2, 4]:
                these_algs = these_algs[0]
            else:
                these_algs = these_algs[1]
        method = these_algs[np.argmax(results[key])]
        #### picking from hyperparameters

        try:
            these_args, all_results = pickle.load(open('interpret_results/results/alg{0}_d{1}_g{2}_t{3}_bw{4}.pkl'.format(method, args.dataset, args.group, t, '1' if args.hetero_bw_api else '0'), 'rb'))
        except:
            continue

        metric_names = ['soft acc', 'soft tpr diff', 'soft tpr diff for A vs PI', 'soft TPR diff for all in API']
        metric_values = [[] for _ in range(len(metric_names))]
        for i in range(len(all_results)):
            y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, mid_indices, granular_indices, nums_ordering = all_results[i]

            if args.test:
                y_this, probs_this = y_test, probs_test
                index_this = 2
            else:
                y_this, probs_this = y_val, probs_val
                index_this = 1

            soft_acc = (y_this*probs_this + (1.-y_this)*(1.-probs_this)).mean()
            metric_values[0].append(soft_acc)

            tprs = []
            y_1 = np.where(y_this==1)[0]
            for g in range(len(granular_indices[index_this])):
                these_indices = np.array(list(set(y_1)&set(granular_indices[index_this][g])))
                if len(these_indices) > 0:
                    tprs.append(np.mean(probs_this[these_indices]))
                else:
                    print("No positives in group {0}")
            tpr_diff = np.amax(tprs)-np.amin(tprs)

            metric_values[1].append(tpr_diff)

            tprs = []
            for g in api_for_mid:
                these_indices = np.array(list(set(y_1)&set(mid_indices[index_this][g])))
                if len(these_indices) > 0:
                    tprs.append(np.mean(probs_this[these_indices]))
                else:
                    print("No positives in group {0}")
            tpr_diff = np.amax(tprs)-np.amin(tprs)
            metric_values[2].append(tpr_diff)


            tprs = []

            for g in api_for_granular:
                these_indices = np.array(list(set(y_1)&set(granular_indices[index_this][g])))
                if len(these_indices) > 0:
                    tprs.append(np.mean(probs_this[these_indices]))
                else:
                    print("No positives in group {0}")
            tpr_diff = np.amax(tprs)-np.amin(tprs)
            metric_values[3].append(tpr_diff)

        yerr = 1.96*np.std(metric_values[to_view])/np.sqrt(len(metric_values[to_view]))
        if t == 0:
            plt.bar([ind+t*.3], [np.mean(metric_values[to_view])], width=.3, edgecolor='k', color='C{}'.format(ind), yerr=yerr) # ADD STD DEV
        elif t == 1:
            plt.bar([ind+t*.3], [np.mean(metric_values[to_view])], width=.3, edgecolor='k', color='C{}'.format(ind), yerr=yerr, hatch='/')
        elif t == 2:
            plt.bar([ind+t*.3], [np.mean(metric_values[to_view])], width=.3, edgecolor='k', color='C{}'.format(ind), yerr=yerr, hatch='o')

        compare[t] = metric_values[to_view]
    for key1 in compare.keys():
        
        print("Alg {0} at tier {1} has {2:.4f}+-{3:.4f}".format(label, key1, np.mean(compare[key1]), 1.96*np.std(compare[key1])/np.sqrt(len(compare[key1]))))
        for key2 in compare.keys():
            if key1 < key2:
                t, p = ttest_ind(compare[key1], compare[key2])
                print("Alg {0} with tiers {1} and {2}, p: {3:.4f}, t: {4:.4f}".format(label, key1, key2, p, t))

fontsize = 11
plt.bar([0], [0], width=0, edgecolor='k', color='w', label='1 Group')
plt.bar([0], [0], width=0, edgecolor='k', color='w', hatch='/', label='2 Groups')
plt.bar([0], [0], width=0, edgecolor='k', color='w', hatch='o', label='{} Groups'.format(len(names[2])-2))
if args.dataset == 0:
    plt.legend(loc='upper left', prop={'size': 8.5})
else:
    plt.legend(prop={'size': 8.5})

plt.ylabel('Max TPR Diff', fontsize=fontsize)
plt.yticks(fontsize=fontsize)


plt.xticks(np.arange(len(algs))+.3, all_algs, fontsize=fontsize)
plt.xlabel('Algorithm', fontsize=fontsize)
plt.tight_layout()
if to_view == 2:
    plt.savefig('interpret_results/images/api_mid.png', dpi=300)
elif to_view == 3:
    plt.savefig('interpret_results/images/api_granular_d{}.png'.format(args.dataset), dpi=300)
plt.close()
print("-----")









