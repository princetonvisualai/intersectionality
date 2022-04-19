from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import torch
from sklearn.model_selection import train_test_split
import copy
import itertools
from copy import deepcopy
from scipy.optimize import linprog
from cvxopt import matrix, solvers
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys

def soft_acc(y, probs):
    return (y*probs + (1.-y)*(1.-probs)).mean()

def reweighing(y, prot_att):
    """Reweighing is a preprocessing technique that Weights the examples in each
    (group, label) combination differently to ensure fairness before
    classification [4]_.

    References:
        .. [4] F. Kamiran and T. Calders,  "Data Preprocessing Techniques for
           Classification without Discrimination," Knowledge and Information
           Systems, 2012.

    notation used from code: https://github.com/Trusted-AI/AIF360/blob/b3f589d3e87afc6ceb19646f426f09e695f81ea6/aif360/algorithms/preprocessing/reweighing.py
    p = privileged protected group
    up = unprivileged protected group
    fav = favorable label condition (let's say label = 1)
    unfav = unfavorable label condition (let's say label = 0)
    """

    """
    inputs: X, y, prot_att
    outputs: weights for the 4 groups, then indices
    """
   
    p = np.where(prot_att == 1)[0]
    up = np.where(prot_att == 0)[0]
    fav = np.where(y == 1)[0]
    unfav = np.where(y == 0)[0]
    
    n = len(y)
    n_p = len(p)
    n_up = len(up)
    n_fav = len(fav)
    n_unfav = len(unfav)

    p_fav = np.array(list(set(p)&set(fav)))
    p_unfav = np.array(list(set(p)&set(unfav)))
    up_fav = np.array(list(set(up)&set(fav)))
    up_unfav = np.array(list(set(up)&set(unfav)))

    n_p_fav = len(p_fav)
    n_p_unfav = len(p_unfav)
    n_up_fav = len(up_fav)
    n_up_unfav = len(up_unfav)

    w_p_fav = n_fav*n_p / (n*n_p_fav)
    w_p_unfav = n_unfav*n_p / (n*n_p_unfav)
    w_up_fav = n_fav*n_up / (n*n_up_fav)
    w_up_unfav = n_unfav*n_up / (n*n_up_unfav)

    return (w_p_fav, w_p_unfav, w_up_fav, w_up_unfav), (p_fav, p_unfav, up_fav, up_unfav)

def error_rates(labels, preds, p_label, up_label, return_all=False):
    # returns (FNR, FPR) of privileged group then unprivileged group
    p = np.where(p_label == 1)[0]
    up = np.where(up_label == 1)[0]

    tn, fp, fn, tp = confusion_matrix(labels[p], preds[p]).ravel()
    p_fnr = fn/(fn+tp)
    p_fpr = fp/(fp+tn)
    if return_all:
        p_tpr = tp/(tp+fn)
        p_tnr = tn/(tn+fp)
        p_all = [p_fpr, p_fnr, p_tpr, p_tnr]

    tn, fp, fn, tp = confusion_matrix(labels[up], preds[up]).ravel()
    up_fnr = fn/(fn+tp)
    up_fpr = fp/(fp+tn)
    if return_all:
        up_tpr = tp/(tp+fn)
        up_tnr = tn/(tn+fp)
        up_all = [up_fpr, up_fnr, up_tpr, up_tnr]

    if return_all:
        return p_all, up_all

    return (p_fnr, p_fpr), (up_fnr, up_fpr)

def max_error_rates(labels, preds, groups, names=None, return_all=False, combo_format=False):
    all_fpr = []
    all_fnr = []
    all_acc = []
    if combo_format:
        combos = groups
    else:
        # groups format should be [[att1_a, att1_b], [att2_a, att2_b]], or something of this nature
        assert len(groups) == 2 # more is not implemented yet
        #groups = [[np.where(groups[0]==1)[0], np.where(groups[0]==0)[0]], [np.where(groups[1]==1)[0], np.where(groups[1]==0)[0]]]
        combos = list(itertools.product(groups[0], groups[1]))
    for i, group in enumerate(combos):
        if combo_format:
            indices = np.where(group==1)[0]
        else:
            one = np.where(group[0] == 1)[0]
            two = np.where(group[1] == 1)[0]
            indices = np.array(list(set(one)&set(two)))
        if np.mean(labels[indices]) in [0, 1]: # don't count FPR or FNR for groups where only has one label, or one prediciton?
            continue
        if len(indices) == 0: # no one in this group
            continue
        tn, fp, fn, tp = confusion_matrix(labels[indices], preds[indices]).ravel()
        fnr = fn/(fn+tp)
        fpr = fp/(fp+tn)
        all_fpr.append(fpr)
        all_fnr.append(fnr)
        all_acc.append(np.mean(labels[indices]==preds[indices]))
    if return_all:
        return all_fpr, all_fnr, all_acc
    return np.amax(all_fpr)-np.amin(all_fpr), np.amax(all_fnr)-np.amin(all_fnr)

#### from reweighting google paper https://arxiv.org/abs/1901.04966###

def reweigh_equalopp(X, y, protected_train, n_iters=100, model_type='lr'):
    def debias_weights(original_labels, predicted, protected_attributes, multipliers):
        exponents = np.zeros(len(original_labels))
        for i, m in enumerate(multipliers):
            exponents -= m * protected_attributes[i] 
        weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
        weights = np.where(original_labels > 0, 1 - weights, weights) # for each group, all 0's get one weight and all 1's get another
        return weights

    def get_error_and_violations(y_pred, y, protected_attributes):
        acc = np.mean(y_pred != y)
        violations = []
        for p in protected_attributes:
            protected_idxs = np.where(np.logical_and(p > 0, y > 0)) 
            positive_idxs = np.where(y > 0)
            violations.append(np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs])) # P(^y=1|y=1) - P(^y=1|y=1, p=1) TPR
        pairwise_violations = []
        for i in range(len(protected_attributes)):
            for j in range(i+1, len(protected_attributes)):
                protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
                if len(protected_idxs[0]) == 0:
                    continue
                pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
        return acc, violations, pairwise_violations # violations are multiplied by learning rate and added to 'multipliers'
    multipliers = np.zeros(len(protected_train))
    weights = np.array([1] * X.shape[0])
    learning_rate = 1.
    for it in range(n_iters):
        if model_type == 'lr':
            model = LogisticRegression()
        elif model_type == 'rf':
            model = RandomForestClassifier()
        elif model_type == 'svm':
            model = SVC()
        else:
            raise NotImplementedError
        model.fit(X, y, weights)
        y_pred = model.predict(X)
        weights = debias_weights(y, y_pred, protected_train, multipliers)
        acc, violations, pairwise_violations = get_error_and_violations(y_pred, y, protected_train)
        multipliers += learning_rate * np.array(violations)
    return model

def jelly_foulds(X, y, protected_train, n_iters, device=torch.device('cpu'), batch_size=64, version=0, lamb=.1):
    pytorch = torch.cuda.is_available()
    order = np.arange(len(X))

    model = MLP(X.shape[1]).to(device) 
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=.005, momentum=0.9)
    for it in range(n_iters):
        all_probs = np.zeros(len(X))
        np.random.shuffle(order)
        for batch_start in np.arange(0, len(X), batch_size):
            tprs = []
            loss = 0.
            optimizer.zero_grad()
            # first figure out which is the max TPR and which is least for this batch
            this_X, this_y = X[order[batch_start:batch_start+batch_size]], y[order[batch_start:batch_start+batch_size]]
            outputs = model(torch.tensor(this_X, requires_grad=True).float().to(device)).squeeze()
            label = torch.tensor(this_y).float().to(device)
            for m in range(len(protected_train)):
                in_group = protected_train[m][order[batch_start:batch_start+batch_size]]
                these_indices = np.array(list(set(np.where(in_group==1)[0])&(set(np.where(this_y==1)[0]))))
                tpr = outputs[these_indices]
                tprs.append(tpr.mean())
            # now actually get the loss
            if pytorch:
                numpy_tprs = [tpr.data.cpu().numpy() for tpr in tprs]
                max_tpr = np.argmax(numpy_tprs)
                min_tpr = np.argmin(numpy_tprs)
            else:
                numpy_tprs = [tpr.data.numpy() for tpr in tprs]
                max_tpr = np.argmax(numpy_tprs)
                min_tpr = np.argmin(numpy_tprs)
            loss = criterion(outputs.squeeze(), label.squeeze()).squeeze().mean()
            loss += lamb*torch.log(tprs[max_tpr]/tprs[min_tpr]) # TODO: maybe add alpha/beta regularizing terms if this is undefined
            loss.backward()
            optimizer.step()
    return model



class MLP(nn.Module):
    def __init__(self, input_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_features, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sig(x)

    def features(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def reweigh_equalopp_mlp(X, y, protected_train, n_iters=100, device=torch.device('cpu'), pytorch_lr=.005, reweigh_lr=1., version=0): # version that is non-convex unlike above, so manipulates a mlp 
    # version 0 is original discrete, version 1 is continuous values otherwise same
    pytorch = torch.cuda.is_available()
    def get_error_and_violations(y_pred, y, protected_attributes):
        acc = np.mean(y_pred != y)
        violations = []
        for p in protected_attributes:
            protected_idxs = np.where(np.logical_and(p > 0, y > 0))
            positive_idxs = np.where(y > 0)
            violations.append(np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs]))
        violations = np.nan_to_num(violations) # so if a group isn't present, it won't err
        pairwise_violations = []
        for i in range(len(protected_attributes)):
            for j in range(i+1, len(protected_attributes)):
                protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
                if len(protected_idxs[0]) == 0:
                    continue
            pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
        return acc, violations, pairwise_violations

    def debias_weights(original_labels, predicted, protected_attributes, multipliers):
        exponents = np.zeros(len(original_labels))

        for i, m in enumerate(multipliers):
            exponents -= m * protected_attributes[i] 
        weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
        weights = np.where(original_labels > 0, 1 - weights, weights) # for each group, all 0's get one weight and all 1's get another
        return weights
    
    order = np.arange(len(X))
    multipliers = np.zeros(len(protected_train))
    weights = np.array([1] * X.shape[0])
    learning_rate = reweigh_lr
    model = MLP(X.shape[1]).to(device)  # might have to do pytorch so can set weights UGH
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=pytorch_lr, momentum=0.9)
    for it in range(n_iters):
        np.random.shuffle(order)
        for batch_start in np.arange(0, len(X), 64):
            optimizer.zero_grad()
            this_weights = weights[order[batch_start:batch_start+64]]
            this_X, this_y = X[order[batch_start:batch_start+64]], y[order[batch_start:batch_start+64]]
            outputs = model(torch.tensor(this_X, requires_grad=True).float().to(device))
            label = torch.tensor(this_y).float().to(device)
            loss = torch.mul(criterion(outputs, label).squeeze(), torch.tensor(this_weights).float().to(device))
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        if pytorch:
            y_probs = model(torch.from_numpy(X).float().to(device)).data.cpu().numpy().squeeze()
        else:
            y_probs = model(torch.from_numpy(X).float()).data.numpy().squeeze()
        if version == 0:
            y_pred = y_probs > .5
        elif version == 1:
            y_pred = y_probs
        weights = debias_weights(y, y_pred, protected_train, multipliers)
        acc, violations, pairwise_violations = get_error_and_violations(y_pred, y, protected_train)
        multipliers += learning_rate * np.array(violations)
    return model


def mlp_train(X_train, y_train, n_iters=100, X_val=None, y_val=None, device=torch.device('cpu'), model=None, lr=.005, batch_size=64, name=''):
    pytorch = torch.cuda.is_available()
    if model is None:
        model = MLP(X_train.shape[1]).to(device) 
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    weights = np.array([1] * X_train.shape[0])
    order = np.arange(len(X_train))

    if X_val is not None:
        all_val_preds = []
        all_train_preds = []
    for it in range(n_iters):
        np.random.shuffle(order)
        train_loss = 0
        correct = 0
        for batch_start in np.arange(0, len(X_train), batch_size):
            optimizer.zero_grad()
            this_weights = weights[order[batch_start:batch_start+batch_size]]
            this_X, this_y = X_train[order[batch_start:batch_start+batch_size]], y_train[order[batch_start:batch_start+batch_size]]
            outputs = model(torch.tensor(this_X, requires_grad=True).float().to(device))
            if pytorch:
                correct += np.sum(outputs.data.cpu().numpy().round().squeeze()==this_y)
            else:
                correct += np.sum(outputs.data.numpy().round().squeeze()==this_y)
            label = torch.tensor(this_y, requires_grad=False).float().to(device)
            loss = torch.mul(criterion(outputs.squeeze(), label.squeeze()).squeeze(), torch.tensor(this_weights).float().to(device))
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_acc = correct / len(X_train)
        if X_val is not None:
            if pytorch:
                preds_val = model(torch.from_numpy(X_val).to(device).float()).data.cpu().numpy().squeeze()
                preds_train = model(torch.from_numpy(X_train).to(device).float()).data.cpu().numpy().squeeze()
            else:
                preds_val = model(torch.from_numpy(X_val).to(device).float()).data.numpy().squeeze()
                preds_train = model(torch.from_numpy(X_train).to(device).float()).data.numpy().squeeze()
            all_val_preds.append(preds_val)
            all_train_preds.append(preds_train)
    if X_val is not None:
        pickle.dump([all_train_preds, all_val_preds], open('./results/{}_try.pkl'.format(name), 'wb'))
    return model



def jelly_mlp_test(all_models, X_test, indices, device, rtn_probs=False):
    pytorch = torch.cuda.is_available()
    preds = np.zeros(len(X_test))
    for i in range(len(indices)):
        these_indices = np.array(list(indices[i]))
        if pytorch:
            these_probs = all_models[i](torch.tensor(X_test[these_indices]).float().to(device)).data.cpu().numpy()
        else:
            these_probs = all_models[i](torch.tensor(X_test[these_indices]).float().to(device)).data.numpy()
        if rtn_probs:
            preds[these_indices] = these_probs.squeeze()
        else:
            these_preds = these_probs.squeeze() > .5
            preds[these_indices] = these_preds
    return preds

## will have to remove all the samples that are not in any of these groups
def jelly_multi_round_test(all_models, X_test, indices, brs):
    preds = np.zeros(len(X_test))
    for i in range(len(indices)):
        these_indices = np.array(list(indices[i]))
        these_probs = all_models[i].predict_proba(X_test[these_indices])[:, 1]
        if brs is None:
            preds[these_indices] = these_probs
        else:
            thresh = np.sort(these_probs)[-int(brs[i]*len(these_indices))-1]
            these_preds = these_probs > thresh
            preds[these_indices] = these_preds
    return preds


def jelly_boost_mlp(X, y, protected_train, p_idx, order, n_iters=100, device=torch.device('cpu'), version=0, lr=.5):  
    # p_idx is the index of protected_train that is the one this model is for (basically tries to create a model where this group has a better tpr than all other groups. this way, it will only use others to become better
    # order is first the least close than closer, etc. each slot is a list, possibly of just one index (index of protected_train)
    # version is what kind of alg this is.  0 is by-group, 1 is by-individual based on correct or incorrect classification
    all_weights = []
    all_tprs = []
    all_tnrs = []
    all_aucs = []
    def get_error_and_violations(y_pred, y, protected_attributes):
        acc = np.mean(y_pred != y)
        violations = []
        this_tprs = []
        this_tnrs = []
        for p in protected_attributes:
            protected_idxs = np.where(np.logical_and(p > 0, y > 0))
            positive_idxs = np.where(np.logical_and(protected_attributes[p_idx]>0, y > 0))
            #positive_idxs = np.where(y > 0)
            if version == 0:
                raise NotImplementedError
                violations.append(np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs])) # P(^y=1|y=1, special=1) - P(^y=1|y=1, p=1) TPR.
            elif version == 1:
                protected_idxs_other = np.where(np.logical_and(p > 0, y < 1))
                positive_idxs_other = np.where(np.logical_and(protected_attributes[p_idx]>0, y < 1))

                this_diff = np.mean(y_pred[protected_idxs])-np.mean(y_pred[protected_idxs_other])
                group_diff = np.mean(y_pred[positive_idxs])-np.mean(y_pred[positive_idxs_other])

                violations.append(group_diff-this_diff) 
            elif version == 2:
                protected_idxs_other = np.where(np.logical_and(p > 0, y < 1))
                positive_idxs_other = np.where(np.logical_and(protected_attributes[p_idx]>0, y < 1))

                this_overlap = overlap(y_pred[protected_idxs_other], y_pred[protected_idxs], 100)
                group_overlap = overlap(y_pred[positive_idxs_other], y_pred[positive_idxs], 100)

                this_diff = np.mean(y_pred[protected_idxs])-np.mean(y_pred[protected_idxs_other])
                group_diff = np.mean(y_pred[positive_idxs])-np.mean(y_pred[positive_idxs_other])

                violations.append(this_diff-group_diff) 
            else:
                raise NotImplementedError
            this_tprs.append(np.mean(y_pred[protected_idxs]))
        pairwise_violations = []
        all_tprs.append(this_tprs)
        all_tnrs.append(this_tnrs)
        for i in range(len(protected_attributes)):
            for j in range(i+1, len(protected_attributes)):
                protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
                if len(protected_idxs[0]) == 0:
                    continue
            pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
        return acc, violations, pairwise_violations

    def debias_weights(original_labels, predicted, protected_attributes, multipliers):
        exponents = np.zeros(len(original_labels))
        for i, m in enumerate(multipliers):
            exponents -= m * protected_attributes[i] 
        weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
        weights = np.where(original_labels > 0, 1 - weights, 1-weights) # for each group, all 0's get one weight and all 1's get another
        return weights

    pytorch = torch.cuda.is_available()
    learning_rate = lr
    order = np.arange(len(X))
    multipliers = np.zeros(len(protected_train))
    weights = np.array([1] * X.shape[0])
    model = MLP(X.shape[1]).to(device)  
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    for it in range(n_iters):
        np.random.shuffle(order)
        for batch_start in np.arange(0, len(X), 64):
            optimizer.zero_grad()
            this_weights = weights[order[batch_start:batch_start+64]]
            this_X, this_y = X[order[batch_start:batch_start+64]], y[order[batch_start:batch_start+64]]
            outputs = model(torch.tensor(this_X, requires_grad=True).float().to(device)).squeeze()
            label = torch.tensor(this_y).float().to(device)
            loss = torch.mul(criterion(outputs, label).squeeze(), torch.tensor(this_weights).float().to(device))
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        if pytorch:
            y_probs = model(torch.from_numpy(X).float().to(device)).data.cpu().numpy().squeeze()
        else:
            y_probs = model(torch.from_numpy(X).float()).data.numpy().squeeze()
        y_pred = y_probs # continuous version
        this_aucs = []
        this_weights = []
        y_0 = np.where(y==0)[0]
        y_1 = np.where(y==1)[0]
        for g in range(len(protected_train)):
            this_indices = np.where(protected_train[g]==1)[0]
            these_weights = weights[this_indices]
            this_weights.append(weights[list(set(y_0)&set(this_indices))[0]])
            this_weights.append(weights[list(set(y_1)&set(this_indices))[0]])
            this_aucs.append(roc_auc_score(y[this_indices], y_probs[this_indices]))
        all_aucs.append(this_aucs)
        all_weights.append(this_weights)
        acc, violations, pairwise_violations = get_error_and_violations(y_pred, y, protected_train)
        violations = np.clip(np.array(violations), None, 0) # weights can only go down for other groups
        multipliers += learning_rate * np.array(violations)
        weights = debias_weights(y, y_pred, protected_train, multipliers)
    pickle.dump([all_weights, all_tprs, all_tnrs, all_aucs], open('try_{}.pkl'.format(p_idx), 'wb'))
    return model


# continuously postprocesses
def jelly_postprocess(probs, y, indices, eps=.05): # ll = .9??
    ratio_version = False
    probs, y = np.array(probs), np.array(y)
    c = []
    G = []
    h = []
    for g in range(len(indices)):
        this_indices = np.array(list(indices[g]))
        this_y, this_p = y[this_indices], probs[this_indices]

        c.append((((2*this_y)-1)*this_p).mean() * (len(this_indices)/len(y)))
        c.append(((2*this_y)-1).mean() * (len(this_indices)/len(y)))

        # constraints that transformed probabilities are between 0 and 1
        min_p, max_p = np.amin(this_p), np.amax(this_p)
        constraint = np.zeros(2*len(indices))
        constraint[2*g] = max_p
        constraint[2*g+1] = 1
        G.append(constraint)
        h.append(1)

        constraint = np.zeros(2*len(indices))
        constraint[2*g] = -min_p
        constraint[2*g+1] = -1
        G.append(constraint)
        h.append(0)

        if ratio_version:
            ll = 1.-eps
            ul = 1./ll

        for g1 in range(len(indices)):
            if g < g1:
                this_indices1 = np.array(list(indices[g1]))
                this_y1, this_p1 = y[this_indices1], probs[this_indices1]

                constraint = np.zeros(2*len(indices))
                if ratio_version:
                    constraint[2*g] = np.sum(this_y*this_p)/np.sum(this_y)
                    constraint[2*g+1] = 1
                    constraint[2*g1] = -ul*np.sum(this_y1*this_p1)/np.sum(this_y)
                    constraint[2*g1+1] = -ul
                    h.append(0)
                else:
                    constraint[2*g] = np.sum(this_y*this_p)/np.sum(this_y)
                    constraint[2*g+1] = np.sum(this_y)/np.sum(this_y)
                    constraint[2*g1] = -np.sum(this_y1*this_p1)/np.sum(this_y1)
                    constraint[2*g1+1] = -np.sum(this_y1)/np.sum(this_y1)
                    h.append(eps)

                G.append(constraint)

                constraint = np.zeros(2*len(indices))
                if ratio_version: # wrong
                    assert NotImplementedError
                    constraint[2*g] = -np.mean(this_y*this_p)
                    constraint[2*g+1] = -np.mean(this_y)
                    constraint[2*g1] = ll*np.mean(this_y1*this_p1)
                    constraint[2*g1+1] = ll*np.mean(this_y1)
                    h.append(0)
                else:
                    constraint[2*g] = -np.sum(this_y*this_p)/np.sum(this_y)
                    constraint[2*g+1] = -np.sum(this_y)/np.sum(this_y)
                    constraint[2*g1] = np.sum(this_y1*this_p1)/np.sum(this_y1)
                    constraint[2*g1+1] = np.sum(this_y1)/np.sum(this_y1)
                    h.append(eps)
                G.append(constraint)
  
    c = -np.array(c)
    c, G, h = matrix(np.array(c).astype(np.double), (2*len(indices), 1), 'd'), matrix(np.array(G)), matrix(np.array(h).astype(np.double), (len(h), 1), 'd')
    sol = solvers.lp(c, G, h)
    # variables to learn are a and b for each group
    vals = np.array(sol['x'])[:, 0]
    #solved_values = np.sum(np.array(G)*np.expand_dims(vals, 0), axis=1)
    return vals




