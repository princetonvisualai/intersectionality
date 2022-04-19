import gerryfair
import numpy as np
import torch
import pickle
import pandas as pd
import os

def train_model(args, X_train, y_train, labels_train, nums_ordering, device, 
                all_indices, X_val, X_test):
    
    if args.fairness_alg==29:
        C, max_iter, gamma = 5, 200, 0.001
    elif args.fairness_alg==30:
        C, max_iter, gamma = 10, 200, 0.005
    elif args.fairness_alg==31:
        C, max_iter, gamma = 20, 100, 0.005
    elif args.fairness_alg==32:
        C, max_iter, gamma = 10, 200, 0.001
    elif args.fairness_alg==33:
        C, max_iter, gamma = 5, 50, 0.001
    elif args.fairness_alg==34:
        C, max_iter, gamma = 10, 100, 0.01

    prot_train = np.zeros((X_train.shape[0], len(labels_train)))
    for i, lab in enumerate(labels_train):
       prot_train[:, i] = lab
     
    prot_train = pd.DataFrame(data=prot_train)
    X_train = pd.DataFrame(data=X_train)
    X_val = pd.DataFrame(data=X_val)
    X_test = pd.DataFrame(data=X_test)
    #y_train = pd.DataFrame(data=y_train)
    fair_model = gerryfair.model.Model(C=C, printflag=False, gamma=gamma, fairness_def='FN', max_iters=max_iter)
    fair_model.train(X_train, prot_train, y_train)
    
    preds = {}

    for name, split in [('train', X_train), ('val', X_val), ('test', X_test)]:
        this_pred = np.array(fair_model.predict(split))
        preds[name] = this_pred
        

    soft_acc = (preds['train']*y_train+(1-preds['train'])*(1-y_train)).mean()
    return preds['train'], preds['val'], preds['test']



