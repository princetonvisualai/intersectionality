import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse

parser = argparse.ArgumentParser(description='intersectionality')
parser.add_argument('--version', type=int, default=0, help='described below')
parser.add_argument('--test', action='store_true', default=False, help='if set true, then test instead of val')
args = parser.parse_args()


version = 0 ## this is where you manipulate number of bf and check when trained on all or bf
version = 1 ## this is where you construct the 5x5 plot to see how it looks
version = 2 ## this is similar to where you manipulate number of bf


def soft_acc(y, probs):
    return (y*probs + (1.-y)*(1.-probs)).mean()

metric_names = ['Acc', 'AUC', 'TPR']
metric = 1

for d in np.arange(5):
    metric_values = [[] for _ in range(len(metric_names))]
    std_values = [[] for _ in range(len(metric_names))]
    print("Dataset {}".format(d))
    if d in [0, 3]:
        method = '1'
    else:
        method = '2'
    if args.version == 0:
        nums = [10, 50, 500, -1]
        for bf_train in range(2):
            graph_nums = []
            for num in nums:
                these_args, all_results = pickle.load(open('interpret_results/results/alg{0}_d{1}_p{2}_n{3}_bf{4}.pkl'.format(method, d, 0, num, bf_train), 'rb'))
                names = all_results[0][9][0]
                these_metrics = [[] for _ in range(len(metric_names))]
                graph_nums.append(len(all_results[0][6][0][2]))
                for i in range(len(all_results)):
                    group_metrics = [[[] for bla in range(len(metric_names)*2)] for _ in range(len(names))]
                    y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, mid_indices, granular_indices, nums_ordering = all_results[i]

                    bf_indices = all_indices[1][2]
                    acc = soft_acc(y_val[bf_indices], probs_val[bf_indices])
                    these_metrics[0].append(acc)
                    these_metrics[1].append(roc_auc_score(y_val[bf_indices], probs_val[bf_indices]))
                    bf_1_indices = np.array(list(set(np.where(y_val==1)[0])&set(bf_indices)))
                    these_metrics[2].append(np.mean(probs_val[bf_1_indices]))
                for m in range(len(metric_names)):
                    metric_values[m].append(np.mean(these_metrics[m]))
                    std_values[m].append(np.std(these_metrics[m]))
                    


        
        for m in range(len(metric_names)):
            plt.errorbar(graph_nums, metric_values[m][:len(nums)], yerr=std_values[m][:len(nums)], label='All {}'.format(metric_names[m]), c='C{}'.format(m))
            plt.errorbar(graph_nums, metric_values[m][len(nums):], yerr=std_values[m][:len(nums)], label='BF {}'.format(metric_names[m]), c='C{}'.format(m), linestyle='dashed')
        plt.xlabel('Number of Black Female')
        plt.ylabel('Evaluation Metric')
        plt.xscale('log')
        plt.legend()
        plt.savefig('./interpret_results/images/bftrain_d{0}.png'.format(d), dpi=200)
        plt.close()
    elif args.version == 1:
        grid = np.zeros((4, 4, 5))

        new_ordering = np.array([2, 0, 3, 1])
        for bf_train in np.arange(0, 4):
            these_args, all_results = pickle.load(open('interpret_results/results/alg{0}_d{1}_p{2}_n{3}_bf{4}.pkl'.format(method, d, 0, -1, bf_train), 'rb'))
            names = all_results[0][9][0]

            for i in range(len(all_results)):
                group_metrics = [[[] for bla in range(len(metric_names)*2)] for _ in range(len(names))]
                y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, mid_indices, granular_indices, nums_ordering = all_results[i]

                for g in range(len(all_indices[2])):
                    these_indices = all_indices[2][g]
                    acc = soft_acc(y_test[these_indices], probs_test[these_indices])
                    auc = roc_auc_score(y_test[these_indices], probs_test[these_indices])
                    these_1_indices = np.array(list(set(np.where(y_test==1)[0])&set(these_indices)))
                    tpr = np.mean(probs_test[these_1_indices])

                    grid[bf_train][g][i] = [acc, auc, tpr][metric]

        grid = grid[new_ordering]
        grid = grid[:, new_ordering]
        ticklabels = np.array([name.replace(' ', '\n') for name in names])
        ticklabels = ticklabels[new_ordering]



        firstchunk = lambda x: '{:.0f}'.format(100.*x)
        secondchunk = lambda x: r'$\pm$ {:.0f}'.format(100.*x)
        firstchunk, secondchunk = np.vectorize(firstchunk), np.vectorize(secondchunk)
        annots = np.core.defchararray.add(firstchunk(np.mean(grid, axis=2)), secondchunk(np.std(grid, axis=2)))
        annots = annots.astype(str)

        font = 15

        do_reverse = True

        if do_reverse:
            sns.heatmap(np.mean(grid, axis=2).T, annot=firstchunk(np.mean(grid, axis=2)).astype(str).T, annot_kws={'va':'bottom', 'fontsize': font+4}, fmt="", cbar=False, cmap='Blues')
            sns.heatmap(np.mean(grid, axis=2).T, annot=secondchunk(np.std(grid, axis=2)).T, annot_kws={'va':'top', 'fontsize': font-3}, fmt="", cbar=False, cmap='Blues')
            ax = sns.heatmap(np.mean(grid, axis=2).T, annot=False, fmt='', xticklabels=ticklabels, yticklabels=ticklabels, annot_kws={'fontsize': font}, cmap='Blues')
        else:
            sns.heatmap(np.mean(grid, axis=2), annot=firstchunk(np.mean(grid, axis=2)).astype(str), annot_kws={'va':'bottom', 'fontsize': font+4}, fmt="", cbar=False, cmap='Blues')
            sns.heatmap(np.mean(grid, axis=2), annot=secondchunk(np.std(grid, axis=2)), annot_kws={'va':'top', 'fontsize': font-3}, fmt="", cbar=False, cmap='Blues')
            ax = sns.heatmap(np.mean(grid, axis=2), annot=False, fmt='', xticklabels=ticklabels, yticklabels=ticklabels, annot_kws={'fontsize': font}, cmap='Blues')
        if do_reverse:
            plt.ylabel('Test', fontsize=font+2)
            plt.xlabel('Train', fontsize=font+2)
        else:
            plt.xlabel('Test', fontsize=font+2)
            plt.ylabel('Train', fontsize=font+2)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = font-1)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = font-1)
        plt.tight_layout()
        plt.savefig('./interpret_results/images/bftrain_{0}_{1}.png'.format(d, 'rev' if do_reverse else 'norev'), dpi=300)
        plt.close()
    elif args.version == 2:
        the_group = 0 # 0 is bf, 1 is wm

        ratios = np.arange(0, 1.1, .1)
        if the_group == 0:
            nums = [0, 5, 10, 50, 100, 200, 400, 500, 1000, 1500, 2000, 3000, -1]
            focus_idx = 2
            constant = 4
        elif the_group == 1:
            nums = [0, 5, 10, 50, 100, 200, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, -1]
            constant = 15
            focus_idx = 1
        to_view = 1 

        best_ratios = []

        graph_nums = []
        for num in [0]:
            comparison_values = [[] for _ in range(len(metric_names))]
            compstd_values = [[] for _ in range(len(metric_names))]
            for bf_train in (range(4, 15) if the_group == 0 else range(15, 26)):
                this_ratio = ratios[bf_train-constant]
                these_args, all_results = pickle.load(open('interpret_results/results/alg{0}_d{1}_p{2}_n{3}_bf{4}.pkl'.format(method, d, 0, num, bf_train), 'rb'))
                names = all_results[0][9][0]
                these_metrics = [[] for _ in range(len(metric_names))]
                if (bf_train == 4 and the_group == 0) or (bf_train == 15 and the_group == 1):
                    graph_nums.append(len(all_results[0][6][0][focus_idx]))
                for i in range(len(all_results)):
                    group_metrics = [[[] for bla in range(len(metric_names)*2)] for _ in range(len(names))]
                    y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, mid_indices, granular_indices, nums_ordering = all_results[i]

                    bf_indices = all_indices[1][focus_idx]
                    acc = soft_acc(y_val[bf_indices], probs_val[bf_indices])
                    these_metrics[0].append(acc)
                    these_metrics[1].append(roc_auc_score(y_val[bf_indices], probs_val[bf_indices]))
                    bf_1_indices = np.array(list(set(np.where(y_val==1)[0])&set(bf_indices)))
                    these_metrics[2].append(np.mean(probs_val[bf_1_indices]))
                for m in range(len(metric_names)):
                    comparison_values[m].append(np.mean(these_metrics[m]))
                    compstd_values[m].append(np.std(these_metrics[m]))

            ##### re-tuning #####
            #these_args, all_results = pickle.load(open('results/alg{0}_d{1}_p{2}_n{3}_bf{4}.pkl'.format(method, d, 0, num, bf_train), 'rb'))
            #these_metrics = [[] for _ in range(len(metric_names))]
            #for i in range(len(all_results)):
            #    y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, mid_indices, granular_indices, nums_ordering = all_results[i]

            #    bf_indices = all_indices[2][2]
            #    acc = soft_acc(y_test[bf_indices], probs_test[bf_indices])
            #    these_metrics[0].append(acc)
            #    these_metrics[1].append(roc_auc_score(y_test[bf_indices], probs_test[bf_indices]))
            #    bf_1_indices = np.array(list(set(np.where(y_test==1)[0])&set(bf_indices)))
            #    these_metrics[2].append(np.mean(probs_test[bf_1_indices]))
            #for m in range(len(metric_names)):
            #    metric_values[m].append(np.mean(these_metrics[m]))
            #    std_values[m].append(1.96*np.std(these_metrics[m])/np.sqrt(len(these_metrics[m])))
            ##### re-tuning #####

        ratio_index = np.argmax(comparison_values[to_view])
        if the_group == 0:
            bf_train = np.arange(4, 15)[ratio_index]
        elif the_group == 1:
            bf_train = np.arange(15, 26)[ratio_index]
        print("Ratio of bf train: ", ratios[bf_train-constant])
        graph_nums = []
        for num in nums:
            try:
                these_args, all_results = pickle.load(open('interpret_results/results/alg{0}_d{1}_p{2}_n{3}_bf{4}.pkl'.format(method, d, 0, num, bf_train), 'rb'))
            except:
                print("Not found for num {}".format(num))
                continue
            names = all_results[0][9][0]
            these_metrics = [[] for _ in range(len(metric_names))]
            graph_nums.append(len(all_results[0][6][0][focus_idx]))
            for i in range(len(all_results)):
                y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, mid_indices, granular_indices, nums_ordering = all_results[i]

                bf_indices = all_indices[2][focus_idx]
                acc = soft_acc(y_test[bf_indices], probs_test[bf_indices])
                these_metrics[0].append(acc)
                these_metrics[1].append(roc_auc_score(y_test[bf_indices], probs_test[bf_indices]))
                bf_1_indices = np.array(list(set(np.where(y_test==1)[0])&set(bf_indices)))
                these_metrics[2].append(np.mean(probs_test[bf_1_indices]))
            for m in range(len(metric_names)):
                metric_values[m].append(np.mean(these_metrics[m]))
                std_values[m].append(1.96*np.std(these_metrics[m])/np.sqrt(len(these_metrics[m])))


        plt.figure(figsize=(6, 2.7))
        comp_vals = np.array([metric_values[to_view][0]]*len(graph_nums))
        comp_stds = np.array([std_values[to_view][0]]*len(graph_nums))
        plt.errorbar(graph_nums, comp_vals, label='BM+WF', linestyle='dashed', C='C1')
        plt.fill_between(graph_nums, comp_vals-comp_stds, comp_vals+comp_stds, alpha=.4, color='C1')

        fontsize=13
        if the_group == 0:
            plt.errorbar(graph_nums, metric_values[to_view], label='BM+WF+BF', c='C0')
            plt.xlabel('Number of Black Female Individuals in Training Set', fontsize=fontsize)
            plt.ylabel('Black Female AUC', fontsize=fontsize)
        elif the_group == 1:
            plt.errorbar(graph_nums, metric_values[to_view], label='BM+WF+WM', c='C0')
            plt.xlabel('Number of White Male Individuals in Training Set', fontsize=fontsize)
            plt.ylabel('White Male AUC', fontsize=fontsize)
        comp_vals = np.array(metric_values[to_view])
        comp_stds = np.array(std_values[to_view])
        plt.fill_between(graph_nums, comp_vals-comp_stds, comp_vals+comp_stds, alpha=.4, color='C0')

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        do_log = True
        logname = 'nolog'
        if do_log:
            plt.xscale('symlog')
            logname = 'log'
        plt.legend(loc='upper left')

        plt.tight_layout()
        if the_group == 0:
            plt.savefig('./interpret_results/images/bftrain_{1}{0}.png'.format(d, logname), dpi=200)
        elif the_group == 1:
            plt.savefig('./interpret_results/images/wmtrain_{1}{0}.png'.format(d, logname), dpi=200)
        plt.close()



