import argparse
import os

parser = argparse.ArgumentParser(description='Measurement')
parser.add_argument('--experiments', nargs='+', type=int, default=[0, 1, 2, 3], help='which of 4 experiments to run, defined in README')
parser.add_argument('--train_models', action='store_true', default=False, help='if set, will train models')
parser.add_argument('--interpret_results', action='store_true', default=False, help='if set, will print results and generate figures')
args = parser.parse_args()

# python3 main.py --train_model --interpret_results

assert len(set(args.experiments).difference(set([0, 1, 2, 3]))) == 0, 'Valid experiments are between 0-3 (inclusive)'

if args.train_models:
    for exp in sorted(set(args.experiments)):
        if exp == 0:
            print("--- Identities to Include: API ---\n")
            os.system('python3 run_experiment.py --experiment 0')
        elif exp == 1:
            print("--- Identities to Include: Other ---\n")
            os.system('python3 run_experiment.py --experiment 1')
        elif exp == 2:
            print("--- Evaluation ---\n")
            os.system('python3 run_experiment.py --experiment 2')
        elif exp == 3:
            print("--- Progressively Smaller Groups ---\n")
            os.system('bash run_trainbf.sh')
        print()

if args.interpret_results:
    for exp in sorted(set(args.experiments)):
        if exp == 0:
            os.system('python3 interpret_results/interpret_api.py --dataset 0 --test')
            os.system('python3 interpret_results/interpret_api.py --dataset 1 --test')
            os.system('python3 interpret_results/interpret_api.py --dataset 2 --test')
            os.system('python3 interpret_results/interpret_api.py --dataset 3 --test')
            os.system('python3 interpret_results/interpret_api.py --dataset 4 --test')
        elif exp == 1:
            os.system('python3 interpret_results/interpret_other.py --dataset 0 --test')
            os.system('python3 interpret_results/interpret_other.py --dataset 1 --test')
            os.system('python3 interpret_results/interpret_other.py --dataset 2 --test')
            os.system('python3 interpret_results/interpret_other.py --dataset 3 --test')
            os.system('python3 interpret_results/interpret_other.py --dataset 4 --test')
        elif exp == 2:
            os.system('python3 interpret_results/interpret_evalrank.py --dataset 0 --test') 
            os.system('python3 interpret_results/interpret_evalrank.py --dataset 1 --test')
            os.system('python3 interpret_results/interpret_evalrank.py --dataset 2 --test')
            os.system('python3 interpret_results/interpret_evalrank.py --dataset 3 --test')
            os.system('python3 interpret_results/interpret_evalrank.py --dataset 4 --test')
            os.system('python3 interpret_results/interpret_evalrank.py --dataset 0 --test --by_version') 
            os.system('python3 interpret_results/interpret_evalrank.py --dataset 1 --test --by_version')
            os.system('python3 interpret_results/interpret_evalrank.py --dataset 2 --test --by_version')
            os.system('python3 interpret_results/interpret_evalrank.py --dataset 3 --test --by_version')
        elif exp == 3:
            os.system('python3 interpret_results/interpret_bftrain.py --version 1 --test') 
            os.system('python3 interpret_results/interpret_bftrain.py --version 2 --test')
        
