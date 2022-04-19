import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from fairness_algs import *

def train_model(args, X_train, y_train, labels_train, nums_ordering, device, att_feats=None, curr_indices=None, X_val=None, y_val=None, name=''):

    model, all_models = None, None
    postproc_eps = -1

    this_X_train_np = X_train
    this_y_train_np = y_train
    args.method = 2

    if args.fairness_alg == 0:
        model = mlp_train(this_X_train_np, this_y_train_np, n_iters=50, device=device, lr=.001)
        args.method = 4
    elif args.fairness_alg == 1:
        model = mlp_train(this_X_train_np, this_y_train_np, n_iters=100, device=device, lr=.001)
        args.method = 4
    elif args.fairness_alg == 2:
        model = mlp_train(this_X_train_np, this_y_train_np, n_iters=150, device=device, lr=.001)
        args.method = 4
    elif args.fairness_alg == 3:
        model = mlp_train(this_X_train_np, this_y_train_np, n_iters=50, device=device)
        args.method = 4
    elif args.fairness_alg == 4:
        model = mlp_train(this_X_train_np, this_y_train_np, n_iters=100, device=device)
        args.method = 4
    elif args.fairness_alg == 5:
        model = mlp_train(this_X_train_np, this_y_train_np, n_iters=150, device=device, lr=.005)
        args.method = 4
    elif args.fairness_alg == 6:
        model = reweigh_equalopp_mlp(this_X_train_np, this_y_train_np, labels_train, n_iters=150, device=device, reweigh_lr=.1, pytorch_lr=.001)
        args.method = 4
    elif args.fairness_alg == 7:
        model = reweigh_equalopp_mlp(this_X_train_np, this_y_train_np, labels_train, n_iters=150, device=device, reweigh_lr=.2, pytorch_lr=.001)
        args.method = 4
    elif args.fairness_alg == 8:
        model = reweigh_equalopp_mlp(this_X_train_np, this_y_train_np, labels_train, n_iters=150, device=device, reweigh_lr=.5, pytorch_lr=.001)
        args.method = 4
    elif args.fairness_alg == 9:
        model = reweigh_equalopp_mlp(this_X_train_np, this_y_train_np, labels_train, n_iters=150, device=device, reweigh_lr=1., pytorch_lr=.001)
        args.method = 4
    elif args.fairness_alg == 10:
        model = reweigh_equalopp_mlp(this_X_train_np, this_y_train_np, labels_train, n_iters=100, device=device, reweigh_lr=.1, pytorch_lr=.001)
        args.method = 4
    elif args.fairness_alg == 11:
        model = reweigh_equalopp_mlp(this_X_train_np, this_y_train_np, labels_train, n_iters=100, device=device, reweigh_lr=.2, pytorch_lr=.001)
        args.method = 4
    elif args.fairness_alg == 12:
        model = reweigh_equalopp_mlp(this_X_train_np, this_y_train_np, labels_train, n_iters=100, device=device, reweigh_lr=.5, pytorch_lr=.001)
        args.method = 4
    elif args.fairness_alg == 13:
        model = reweigh_equalopp_mlp(this_X_train_np, this_y_train_np, labels_train, n_iters=100, device=device, reweigh_lr=1., pytorch_lr=.001)
        args.method = 4
    elif args.fairness_alg == 20: 
        model = jelly_foulds(this_X_train_np, this_y_train_np, labels_train, n_iters=200, device=device, batch_size=1024, lamb=.01)
        args.method = 4
    elif args.fairness_alg == 21: 
        model = jelly_foulds(this_X_train_np, this_y_train_np, labels_train, n_iters=200, device=device, batch_size=1024, lamb=.05)
        args.method = 4
    elif args.fairness_alg == 22: 
        model = jelly_foulds(this_X_train_np, this_y_train_np, labels_train, n_iters=200, device=device, batch_size=1024, lamb=.1)
        args.method = 4
    elif args.fairness_alg == 23: 
        model = jelly_foulds(this_X_train_np, this_y_train_np, labels_train, n_iters=250, device=device, batch_size=1024, lamb=.01)
        args.method = 4
    elif args.fairness_alg == 24: 
        model = jelly_foulds(this_X_train_np, this_y_train_np, labels_train, n_iters=250, device=device, batch_size=1024, lamb=.05)
        args.method = 4
    elif args.fairness_alg == 25: 
        model = jelly_foulds(this_X_train_np, this_y_train_np, labels_train, n_iters=250, device=device, batch_size=1024, lamb=.1)
        args.method = 4
    elif args.fairness_alg == 26: 
        model = jelly_foulds(this_X_train_np, this_y_train_np, labels_train, n_iters=300, device=device, batch_size=1024, lamb=.01)
        args.method = 4
    elif args.fairness_alg == 27: 
        model = jelly_foulds(this_X_train_np, this_y_train_np, labels_train, n_iters=300, device=device, batch_size=1024, lamb=.05)
        args.method = 4
    elif args.fairness_alg == 28: 
        model = jelly_foulds(this_X_train_np, this_y_train_np, labels_train, n_iters=300, device=device, batch_size=1024, lamb=.1)
        args.method = 4
    else:
        raise NotImplementedError
    return model, all_models, postproc_eps
