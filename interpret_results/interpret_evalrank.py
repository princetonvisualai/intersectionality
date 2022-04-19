import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns
import pickle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
import copy
from scipy.stats import kendalltau, combine_pvalues
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

sys.path.append('./import_files')
from fairness_algs import *
from all_algs import *

def soft_acc(y, probs):
    return (y*probs + (1.-y)*(1.-probs)).mean()

parser = argparse.ArgumentParser(description='intersectionality')
parser.add_argument('--dataset', type=int, default=0, help='which of 5 folktable datasets from 2018')
parser.add_argument('--test', action='store_true', default=False, help='if set true, then test instead of val')
parser.add_argument('--by_version', action='store_true', default=False, help='if set true, then treat dataset as version and do it by dataset')
args = parser.parse_args()

print(args)

to_view = 1
if args.by_version:
    num_versions = 5
else:
    num_versions = 4
alg_names = []

results, algs = pickle.load(open('interpret_results/hyperparameters/v2.pkl', 'rb'))

all_algs = ['Baseline', 'RWT', 'RDC', 'LOS', 'GRP', 'GRY']

grid = np.zeros((num_versions, len(algs)))
annots = np.chararray((num_versions, len(algs)), itemsize=60)
annots_std = np.zeros((num_versions, len(algs)))
annots_below = np.chararray((num_versions, len(algs)), itemsize=60)

annots_tau = np.zeros((num_versions, len(algs), 2))
annots_fair = np.zeros((num_versions, len(algs), 2))
for ind, label in enumerate(all_algs):
    metric_names = ['acc', 'auc', 'tpr']
    alg_names.append(label)
    compare = {}
    for version in range(num_versions):

        #### picking from hyperparameters
        if args.by_version:
            key = '{0}-d{1}-v{2}'.format(label, version, args.dataset)
        else:
            if version == 0:
                continue
            key = '{0}-d{1}-v{2}'.format(label, args.dataset, version)
        these_algs = algs[label]
        if label in ['RWT']:
            if args.by_version:
                if version in [1, 2, 4]:
                    these_algs = these_algs[0]
                else:
                    these_algs = these_algs[1]
            else:
                if args.dataset in [1, 2, 4]:
                    these_algs = these_algs[0]
                else:
                    these_algs = these_algs[1]
        try:
            method = these_algs[np.argmax(results[key])]
        except:
            annot = '---'
            annots[version, ind] = annot
            print(key)
            continue
        if np.amax(results[key]) == -1:
            print("Not finished running: {}".format(key))
            annot = '---'
            annots[version, ind] = annot
            continue

        #### picking from hyperparameters
        taus = []
        ps = []
        if args.by_version:
            these_args, all_results = pickle.load(open('interpret_results/results/evalrank_alg{0}_d{1}_v{2}.pkl'.format(method, version, args.dataset), 'rb'))
        else:
            these_args, all_results = pickle.load(open('interpret_results/results/evalrank_alg{0}_d{1}_v{2}.pkl'.format(method, args.dataset, version), 'rb'))
        bf_low = []
        wm_high = []
        fairness = []
        for i in range(len(all_results)):
            y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, nums_ordering = all_results[i]

            if args.test:
                y_this, probs_this = y_test, probs_test
                index_this = 2
            else:
                y_this, probs_this = y_val, probs_val
                index_this = 1

            base_rates = []
            tprs = []
            y_1 = np.where(y_this==1)[0]
            for g in range(len(all_indices[index_this])):
                these_indices = np.array(list(set(y_1)&set(all_indices[index_this][g])))
                base_rates.append(np.mean(y_this[all_indices[index_this][g]]))
                if len(these_indices) > 0:
                    tprs.append(np.mean(probs_this[these_indices]))
                else:
                    print("No positives in group {0}".format(g))
            fairness.append(np.amax(tprs)-np.amin(tprs))
            
            bf_low.append(list(np.argsort(tprs)).index(3))
            wm_high.append(list(np.argsort(tprs)).index(0))
            tau, p = kendalltau(base_rates, tprs)
            taus.append(tau)
            ps.append(p)
        p = combine_pvalues(ps)[1]
        grid[version, ind] = np.mean(taus)

        annots_fair[version, ind, 0] = np.mean(fairness)
        annots_fair[version, ind, 1] = np.std(fairness)
        if p < .05:
            annots_tau[version, ind, 0] = np.mean(taus)
            annots_tau[version, ind, 1] = 1.96*np.std(taus)/np.sqrt(len(taus))
        else:
            annots_tau[version, ind, 0] = -1
            annots_tau[version, ind, 1] = -1

if not args.by_version:
    annots_fair, annots_tau = annots_fair[1:], annots_tau[1:]
    grid = grid[1:]
if args.by_version:
    yticklabels = ['ACSIncome', 'ACSPublicCoverage', 'ACSMobility', 'ACSEmployment', 'ACSTravelTime']
else:
    yticklabels = ['Race x\nDisability', 'Marital Status x\nSex', 'Marital Status x\nSex x\nDisability']


if args.by_version and args.dataset == 3:
    annots_fair, annots_tau = annots_fair[0:3], annots_tau[0:3]
    grid = grid[0:3]
    yticklabels = yticklabels[0:3]

firstchunk = lambda x: '{:.0f}'.format(100.*x) if x >= 0 else ''
firstchunk_n = lambda x: '\nF: {:.0f}'.format(100.*x) if x >= 0 else ''
secondchunk = lambda x: r'$\pm${:.0f}'.format(100.*x) if x >= 0 else ''
newline = lambda x: '\n'
newlines = lambda x: '\n\n\n'

firstchunk, secondchunk = np.vectorize(firstchunk), np.vectorize(secondchunk)
firstchunk_n, newline = np.vectorize(firstchunk_n), np.vectorize(newline)
annots_fair_disp = np.core.defchararray.add(firstchunk_n(annots_fair[:, :, 0]), secondchunk(annots_fair[:, :, 1]))
annots_tau_disp = np.core.defchararray.add(np.core.defchararray.add(firstchunk(annots_tau[:, :, 0]), secondchunk(annots_tau[:, :, 1])), newline(annots_tau[:, :, 0]))
annots_fair_disp = annots_fair_disp.astype(str)
annots_tau_disp = annots_tau_disp.astype(str)


if args.test and args.dataset == 0 and not args.by_version: # ACSIncome with race x disp to show difference
    graph_vers = 3 # 0 is bar, 1 is line thing, 2 is same line diff axes, 3 is bars but diff graphs

    version = 0
    taus_with_stat_p = 100.*annots_tau[version, :, 0]
    stds_with_stat_p = 100.*annots_tau[version, :, 1]
    taus_with_stat_p[1] = 0
    stds_with_stat_p[1] = 0
    order = np.argsort(annots_fair[version, :, 0])

    if graph_vers == 0:
        fig, ax1 = plt.subplots(1, figsize=(8, 4))
        ax2 = ax1.twinx()
    elif graph_vers == 1:
        plt.figure(figsize=(6, 3))
        fig, axs = plt.subplots(2)
    elif graph_vers == 2:
        plt.figure(figsize=(6, 3))
    elif graph_vers == 3:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    for ind in range(len(all_algs)):
        if graph_vers == 0:
            ax2.bar(7+ind, taus_with_stat_p[order[ind]], yerr=stds_with_stat_p[order[ind]], color='C{}'.format(order[ind]), hatch='/')
            ax1.bar(ind, 100.*annots_fair[version, :, 0][order[ind]], yerr=100.*annots_fair[version, :, 1][order[ind]], color='C{}'.format(order[ind]))
        elif graph_vers == 1:
            alpha = .8
            linewidth = 4
            capsize = 3
            capthick = 3
            elinewidth=2
            axs[0].errorbar([taus_with_stat_p[ind]], [-.1*ind], xerr = stds_with_stat_p[ind], alpha=alpha, linewidth=linewidth, elinewidth=elinewidth, capsize=capsize, capthick=capthick, c='C{}'.format(ind))
            axs[0].scatter([taus_with_stat_p[ind]], [-.1*ind], s=10, c='C{}'.format(ind))
            axs[1].errorbar([100.*annots_fair[version, :, 0][ind]], [-.1*ind], xerr = 100.*annots_fair[version, :, 1][ind], alpha=alpha, linewidth=linewidth, elinewidth=elinewidth, capsize=capsize, capthick=capthick, c='C{}'.format(ind))
            axs[1].scatter([100.*annots_fair[version, :, 0][ind]], [-.1*ind], s=10, c='C{}'.format(ind))
        elif graph_vers == 2:
            linewidth = 4
            capsize = 3
            capthick = 3
            elinewidth=2
            alpha = .8
            plt.errorbar([taus_with_stat_p[ind]], [100.*annots_fair[version, ind, 0]], xerr=stds_with_stat_p[ind], yerr=100.*annots_fair[version, ind, 1], linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth, c='C{}'.format(ind), alpha=alpha)
        elif graph_vers == 3:
            axs[1].bar(ind, taus_with_stat_p[order[ind]], yerr=stds_with_stat_p[order[ind]], color='C{}'.format(order[ind]))
            axs[0].bar(ind, 100.*annots_fair[version, :, 0][order[ind]], yerr=100.*annots_fair[version, :, 1][order[ind]], color='C{}'.format(order[ind]))


    if graph_vers == 0:
        all_algs[0] = 'Base\nline'
        all_algs[0] = '    Baseline'
        plt.xticks(np.arange(len(all_algs)*2+1), list(np.array(all_algs)[order]) + [''] + list(np.array(all_algs)[order]), fontsize=8.3)
        plt.bar([0], [0], color='w', edgecolor='k', label="Max TPR Diff")
        plt.bar([0], [0], color='w', edgecolor='k', label="Kendall's Tau", hatch='/')
        plt.xlabel('Algorithm')
        ax1.set_ylabel('Max TPR Diff')
        ax2.set_ylabel("Kendall's Tau")
        plt.legend()
        plt.title("(c) ACSIncome: Race x Disability")
    elif graph_vers == 1:
        for ax in [0, 1]:
            axs[ax].set_yticks(-.1*np.arange(len(all_algs)))
            axs[ax].set_yticklabels(all_algs)
            axs[ax].set_ylabel('Algorithm')
        axs[0].set_xlabel("Kendall's Tau")
        axs[1].set_xlabel("Max TPR Diff")
        plt.title("(c) ACSIncome: Race x Disability")
    elif graph_vers == 2:
        plt.xlabel("Kendall's Tau")
        plt.ylabel("Max TPR Diff")
        plt.title("(c) ACSIncome: Race x Disability")
    elif graph_vers == 3:
        all_algs[0] = '    Baseline'
        for ax in [0, 1]:
            axs[ax].set_xlabel('Algorithm')
            axs[ax].set_xticks(np.arange(len(all_algs)))
            axs[ax].set_xticklabels(np.array(all_algs)[order], fontsize=8.3)
        axs[0].set_ylabel('Max TPR Diff')
        axs[1].set_ylabel("Kendall's Tau")
        plt.suptitle("(c) ACSIncome: Race x Disability", y=1)
    plt.tight_layout()
    plt.savefig('./interpret_results/images/evalrank_diff_d0_v1.png', dpi=240)
    plt.close()

plt.figure(figsize=(6, 2.6))
font = 12

# option 1: to just show ranking
#sns.heatmap(grid, annot=annots_tau_disp, annot_kws={'va': 'center', 'fontsize': font}, fmt="", cbar=False, cmap='Blues', vmin=0., vmax=1.)

# option 2: to just show fairness
# to also show the fairness AND ranking
#sns.heatmap(grid, annot=annots_tau_disp, annot_kws={'va':'bottom', 'fontsize': font}, fmt="", cbar=False, cmap='Blues')
#sns.heatmap(grid, annot=annots_fair_disp, annot_kws={'va':'top', 'fontsize': font}, fmt="", cbar=False, cmap='Blues')

# option 3: to just show ranking, but std dev on new line
std_dev = np.core.defchararray.add(newline(annots_tau[:, :, 0]), secondchunk(annots_tau[:, :, 1]))
#std_dev = np.core.defchararray.add(secondchunk(annots_tau[:, :, 1]), newlines(annots_tau[:, :, 1]))
sns.heatmap(grid, annot=firstchunk(annots_tau[:, :, 0]).astype(str), annot_kws={'va': 'baseline', 'fontsize': font}, fmt='', cbar=False, cmap='Blues', vmin=0., vmax=1.)
sns.heatmap(grid, annot=std_dev.astype(str), annot_kws={'va': 'center', 'fontsize': font-5}, fmt="", cbar=False, cmap='Blues', vmin=0., vmax=1.)

ax = sns.heatmap(grid, annot=None, fmt='', yticklabels=yticklabels, xticklabels=alg_names, annot_kws={'fontsize': font}, cmap='Blues', cbar_kws={'label': "Kendall's Tau"}, vmin=0., vmax=1.)

for i in range(len(grid)+1):
    ax.axhline(i, color='white', lw=10)

plt.xlabel('Algorithm')
plt.yticks(rotation=50) 

if args.by_version:
    yticklabels = ['Constrained Race x Sex', 'Race x Disability', 'Marital Status x Sex', 'Marital Status x Sex x Disability']
    plt.title('(b) Protected Attributes: {}'.format(yticklabels[args.dataset]))
    plt.ylabel('Dataset')
    plt.tight_layout()
    plt.savefig('./interpret_results/images/evalrank_v{}.png'.format(args.dataset), dpi=240)
else:
    yticklabels = ['ACSIncome', 'ACSPublicCoverage', 'ACSMobility', 'ACSEmployment', 'ACSTravelTime']
    plt.title('(a) Dataset: {}'.format(yticklabels[args.dataset]))
    plt.ylabel('Protected Attributes')
    plt.tight_layout()
    plt.savefig('./interpret_results/images/evalrank_d{}.png'.format(args.dataset), dpi=240)
plt.close()


