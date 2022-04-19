import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='intersectionality')
parser.add_argument('--experiment', type=int, default=0, help='0 is hetero - api, 1 is hetero - other, 2 is evalranks')
args = parser.parse_args()


bas_methods = [0, 1, 2, 3, 4, 5] # baseline methods
rwt_methods = [6, 7, 8, 9, 10, 11, 12, 13]
rdc_methods = [14, 15, 16, 17, 18, 19] 
los_methods = [20, 21, 22, 23, 24, 25, 26, 27, 28] 
grp_methods = ['eo_exp_plugin']
gry_methods = [29, 30, 31, 32, 33, 34]

methods = bas_methods+rwt_methods+rdc_methods+los_methods+grp_methods+gry_methods

f = open("run_{0}.sh".format(args.experiment), "w")
for m, method in enumerate(methods):

    if method in [6, 7, 8, 9]:
        f.write("for d in 1 2 4\ndo\n")
    elif method in [10, 11, 12, 13]:
        f.write("for d in 0 3\ndo\n")
    else:
        f.write("for d in 0 1 2 3 4\ndo\n")
    
    ### for API hetero experiment ###
    if args.experiment == 0:
        if method in ['eo_exp_plugin']:
            f.write("    python3 groupfair_train_script.py --dataset $d --num_trials 5 --hetero_bw_api --group 0 --tier 0 --expfile {0}\n".format(method))
            f.write("    python3 groupfair_train_script.py --dataset $d --num_trials 5 --hetero_bw_api --group 0 --tier 1 --expfile {0}\n".format(method))
            f.write("    python3 groupfair_train_script.py --dataset $d --num_trials 5 --hetero_bw_api --group 0 --tier 2 --expfile {0}\n".format(method))
        else:
            f.write("    python3 heterogeneity.py --dataset $d --num_trials 5 --fairness_alg {} --hetero_bw_api --group 0 --tier 0\n".format(method))
            f.write("    python3 heterogeneity.py --dataset $d --num_trials 5 --fairness_alg {} --hetero_bw_api --group 0 --tier 1\n".format(method))
            f.write("    python3 heterogeneity.py --dataset $d --num_trials 5 --fairness_alg {} --hetero_bw_api --group 0 --tier 2\n".format(method))
    elif args.experiment == 1:
        ### for other hetero experiment ###
        if method in ['eo_exp_plugin']:
            f.write("    python3 groupfair_train_script.py --dataset $d --num_trials 5 --group 2 --tier 1 --other_exp 1 --expfile {0}\n".format(method))
            f.write("    python3 groupfair_train_script.py --dataset $d --num_trials 5 --group 2 --tier 1 --other_exp 3 --expfile {0}\n".format(method))
        else:
            f.write("    python3 heterogeneity.py --dataset $d --num_trials 5 --fairness_alg {} --group 2 --tier 1 --other_exp 1\n".format(method))
            f.write("    python3 heterogeneity.py --dataset $d --num_trials 5 --fairness_alg {} --group 2 --tier 1 --other_exp 2\n".format(method))
            f.write("    python3 heterogeneity.py --dataset $d --num_trials 5 --fairness_alg {} --group 2 --tier 1 --other_exp 3\n".format(method))
    elif args.experiment == 2:
        ## for the evaluation rank experiment ##
        if method in ['eo_exp_plugin']:
            f.write("    python3 groupfair_train_script.py --dataset $d --num_trials 5 --version 0 --evalrank --expfile {0}\n".format(method))
            f.write("    python3 groupfair_train_script.py --dataset $d --num_trials 5 --version 1 --evalrank --expfile {0}\n".format(method))
            f.write("    python3 groupfair_train_script.py --dataset $d --num_trials 5 --version 2 --evalrank --expfile {0}\n".format(method))
            f.write("    python3 groupfair_train_script.py --dataset $d --num_trials 5 --version 3 --evalrank --expfile {0}\n".format(method))
        else:
            f.write("    python3 eval_rank.py --dataset $d --num_trials 5 --fairness_alg {} --version 0\n".format(method))
            f.write("    python3 eval_rank.py --dataset $d --num_trials 5 --fairness_alg {} --version 1\n".format(method))
            f.write("    python3 eval_rank.py --dataset $d --num_trials 5 --fairness_alg {} --version 2\n".format(method))
            f.write("    python3 eval_rank.py --dataset $d --num_trials 5 --fairness_alg {} --version 3\n".format(method))

    f.write("done\n\n")
f.write("python3 pick_hyperparameters.py --experiment {}".format(args.experiment))
f.close()

os.system("bash run_{}.sh".format(args.experiment))


