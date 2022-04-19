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
from folktables import ACSIncomeDemRank, ACSPublicCoverageDemRank, ACSTravelTimeDemRank, ACSMobilityDemRank, ACSEmploymentFilteredDemRank # 5 from paper, with demographic of races and more
import time
import sys

sys.path.append('./import_files')
from fairness_algs import *
from all_algs import *
from gerryfair_train import train_model as train_model_gry
from fairlearn_train import train_model as train_model_rdc

def create_dataset(args):
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir='Data')
    states = ['CA']
    acs_data = data_source.get_data(states=states, download=True)

    rac1p_names = ['White', 'Black', 'American Indian', 'Alaska Native', 'American Indian and Alaska Native', 'Asian', 'Native Hawaiian and Other Pacific Islander', 'Other', 'Two or more']


    min_samples = 300
    min_positives = 30
    min_negatives = 30

    datasets = [ACSIncomeDemRank, ACSPublicCoverageDemRank, ACSMobilityDemRank, ACSEmploymentFilteredDemRank, ACSTravelTimeDemRank]
    dataset = datasets[args.dataset]

    all_prot = ['DIS', 'MAR', 'RAC1P', 'SEX']

    features, label, group = dataset.df_to_numpy(acs_data)
    features = pd.DataFrame(features, columns=dataset.features)
    feature_names = list(dataset.features)
    features = features.loc[:,~features.columns.duplicated()]

    cont_feats = ['AGEP', 'SCHL', 'WKHP', 'PINCP', 'JWMNP', 'POVPIP'] # could also be discretized
    cat_feats = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'ESR', 'ST', 'FER', 'GCL', 'POWPUMA', 'JWTR', 'PUMA', 'RAC1P', 'RAC2P', 'RAC3P']

    remove_indices = []
    prot_group_names = []
    drop_features = []
    att1, att2, att3 = None, None, None

    if args.version == 0:
        for rac1p in range(len(rac1p_names)):
            these_indices = np.where(feature_names.index('RAC1P') == rac1p+1)[0]
            if rac1p not in [0, 1]:
                remove_indices.append(these_indices)
                continue
            for sex in np.unique(np.array(features['SEX'])):
                prot_group_names.append("{0:.0f}-{1:.0f}".format(rac1p+1, sex))
        drop_features = ['RAC2P', 'RAC3P', 'DIS', 'MAR']
        att1 = 'RAC1P'
        att2 = 'SEX'
    elif args.version == 1:
        att1 = 'RAC1P'
        att2 = 'DIS'
        drop_features = ['RAC2P', 'RAC3P', 'MAR']
    elif args.version == 2:
        att1 = 'MAR'
        att2 = 'SEX'
        drop_features = ['RAC2P', 'RAC3P', 'DIS']
    elif args.version == 3:
        att1 = 'MAR'
        att2 = 'SEX'
        att3 = 'DIS'
        drop_features = ['RAC2P', 'RAC3P']
        for this_att1 in np.unique(np.array(features[att1])):
            att1_indices = np.where(np.array(features[att1])==this_att1)[0]
            for this_att2 in np.unique(np.array(features[att2])):
                att2_indices = np.where(np.array(features[att2])==this_att2)[0]
                for this_att3 in np.unique(np.array(features[att3])):
                    att3_indices = np.where(np.array(features[att3])==this_att3)[0]
                    these_indices = np.array(list(set(att1_indices)&set(att2_indices)&set(att3_indices)))
                    these_labels = label[these_indices]
                    if len(these_indices) < min_samples or np.sum(these_labels) <= min_positives and np.sum(these_labels==0) <= min_negatives:
                        remove_indices.append(these_indices)
                    else:
                        prot_group_names.append('{0:.0f}-{1:.0f}-{2:.0f}'.format(this_att1, this_att2, this_att3))

    if args.version in [1, 2]:
        for this_att1 in np.unique(np.array(features[att1])):
            att1_indices = np.where(np.array(features[att1])==this_att1)[0]
            for this_att2 in np.unique(np.array(features[att2])):
                att2_indices = np.where(np.array(features[att2])==this_att2)[0]
                these_indices = np.array(list(set(att1_indices)&set(att2_indices)))
                these_labels = label[these_indices]
                if len(these_indices) < min_samples or np.sum(these_labels) <= min_positives and np.sum(these_labels==0) <= min_negatives:
                    remove_indices.append(these_indices)
                else:
                    prot_group_names.append('{0:.0f}-{1:.0f}'.format(this_att1, this_att2))

    if len(remove_indices) > 0:
        remove_indices = np.concatenate(remove_indices)
        keep_indices = np.array(list(set(np.arange(len(features))).difference(set(remove_indices))))
        features, label = features.iloc[keep_indices], label[keep_indices]

    nums_ordering = [prot_group_names] # this might be wrong

    these_cat_feats = []
    for feat in features.keys():
        if feat in cat_feats:
            these_cat_feats.append(feat)

    no_race_feats = list(set(these_cat_feats).difference(set([att1, att2, att3])))
    data_cat = features[these_cat_feats]
    data_cat = pd.get_dummies(data_cat.astype(int).astype(str), drop_first=False)
    features = pd.concat([features.drop(no_race_feats,axis=1), data_cat], axis=1)

    features = features.loc[:,~features.columns.duplicated()]

    return features, label, rac1p_names, states, nums_ordering, [att1, att2, att3]

def split_dataset(args, features, label, rac1p_names, states, nums_ordering, atts):
    att1, att2, att3 = atts
    mid_X_train, X_test, mid_y_train, y_test = train_test_split(features, label, test_size=.3)
    X_train, X_val, y_train, y_val = train_test_split(mid_X_train, mid_y_train, test_size=.3)

    labels_train = []
    all_indices = []

    for x, X in enumerate([X_train, X_val, X_test]):
        this_indices = []
        for this_att1 in np.unique(np.array(X[att1])):
            att1_indices = np.where(np.array(X[att1])==this_att1)[0]
            for this_att2 in np.unique(np.array(X[att2])):
                att2_indices = np.where(np.array(X[att2])==this_att2)[0]
                these_indices = np.array(list(set(att1_indices)&set(att2_indices)))
                name = '{0:.0f}-{1:.0f}'.format(this_att1, this_att2)

                if att3 is not None:
                    for this_att3 in np.unique(np.array(X[att3])):
                        att3_indices = np.where(np.array(X[att3])==this_att3)[0]
                        these_indices = np.array(list(set(att1_indices)&set(att2_indices)&set(att3_indices)))
                        name = '{0:.0f}-{1:.0f}-{2:.0f}'.format(this_att1, this_att2, this_att3)
                        if name in nums_ordering[0]:
                            this_indices.append(these_indices)
                            if x == 0:
                                indicator = np.zeros(len(X))
                                indicator[these_indices] = 1
                                labels_train.append(indicator)
                else:
                    if name in nums_ordering[0]:
                        this_indices.append(these_indices)
                        if x == 0:
                            indicator = np.zeros(len(X))
                            indicator[these_indices] = 1
                            labels_train.append(indicator)
        all_indices.append(this_indices)

    min_max_scaler = preprocessing.MinMaxScaler()
    if None in atts:
        atts = atts[:2]
    for x, X in enumerate([X_train, X_val, X_test]):
        X = X.drop(atts,axis=1)
        if x == 0:
            min_max_scaler.fit(X)
            X_train = min_max_scaler.transform(X)
        elif x == 1:
            X_val = min_max_scaler.transform(X)
        elif x == 2:
            X_test = min_max_scaler.transform(X)

    # some sanity checks
    assert len(all_indices) == 3
    assert len(all_indices[0]) == len(all_indices[1])
    assert len(nums_ordering[0]) == len(labels_train)
    assert len(nums_ordering[0]) == len(all_indices[0])
    assert len(all_indices[0]) == len(all_indices[2])
    assert len(labels_train) == len(all_indices[0])
    assert np.sum(labels_train[2]) == len(all_indices[0][2])
    if len(labels_train) > 6:
        assert np.sum(labels_train[6]) == len(all_indices[0][6])

    return X_train, y_train, X_val, y_val, X_test, y_test, all_indices, nums_ordering, labels_train

def main():
    parser = argparse.ArgumentParser(description='intersectionality')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--num_trials', type=int, default=5, help='number of trials')
    parser.add_argument('--dataset', type=int, default=0, help='which of 5 folktable datasets from 2018')
    parser.add_argument('--version', type=int, default=0, help='prot att version, 0 is [Black, white] x [male, female], 1 is RAC1P x DIS, 2 is MAR x SEX, 3 is MAR x SEX x DIS')
    parser.add_argument('--fairness_alg', type=int, default=0, help='all of the different methods we have')
    parser.add_argument('--warnings', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    start = time.time()

    datasets = [ACSIncomeDemRank, ACSPublicCoverageDemRank, ACSMobilityDemRank, ACSEmploymentFilteredDemRank, ACSTravelTimeDemRank]

    if args.seed == -1:
        args.seed = np.random.randint(0, 1000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print("Random seed: {}".format(args.seed))

    features, label, rac1p_names, states, nums_ordering, atts = create_dataset(args)

    all_results = []

    for i in range(args.num_trials):

        X_train, y_train, X_val, y_val, X_test, y_test, all_indices, nums_ordering, labels_train = split_dataset(args, features, label, rac1p_names, states, nums_ordering, atts)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if args.fairness_alg in [14, 15, 16, 17, 18, 19, 29, 30, 31, 32, 33, 34]:
            if args.fairness_alg in [14, 15, 16, 17, 18, 19]:
                probs_train, probs_val, probs_test = train_model_rdc(args, X_train, y_train, labels_train, nums_ordering, device, all_indices, X_val, X_test)
            elif args.fairness_alg in [29, 30, 31, 32, 33, 34]:
                probs_train, probs_val, probs_test = train_model_gry(args, X_train, y_train, labels_train, nums_ordering, device, all_indices, X_val, X_test)
            else:
                raise NotImplementedError
        else:

            model, all_models, postproc_eps = train_model(args, X_train, y_train, labels_train, nums_ordering, device, att_feats=None, curr_indices=all_indices, X_val=None, y_val=None, name='{0}-{1}-{2}'.format(args.fairness_alg, i, args.dataset))

            if args.method in [0]:
                probs_train = model.predict_proba(X_train)[:, 1]
                probs_val = model.predict_proba(X_val)[:, 1]
                probs_test = model.predict_proba(X_test)[:, 1]
            elif args.method in [4]:
                probs_train = model(torch.tensor(X_train).float().to(device)).data.cpu().numpy().squeeze()
                probs_val = model(torch.tensor(X_val).float().to(device)).data.cpu().numpy().squeeze()
                probs_test = model(torch.tensor(X_test).float().to(device)).data.cpu().numpy().squeeze()
            else:
                probs_train = jelly_mlp_test(all_models, X_train, all_indices[0], device, rtn_probs=True)
                probs_val = jelly_mlp_test(all_models, X_val, all_indices[1], device, rtn_probs=True)
                probs_test = jelly_mlp_test(all_models, X_test, all_indices[2], device, rtn_probs=True)

            ## postprocess if exists
            #if postproc_eps >= 0:
            #    learned_vals = jelly_postprocess(probs_train, y_train, all_indices[0], eps=postproc_eps)
            #    for ind in range(len(all_indices[1])):
            #        probs_val[np.array(list(all_indices[1][ind]))] = np.clip(probs_val[np.array(list(all_indices[1][ind]))]*learned_vals[2*ind]+learned_vals[2*ind+1], 0, 1)
            #        probs_test[np.array(list(all_indices[2][ind]))] = np.clip(probs_test[np.array(list(all_indices[2][ind]))]*learned_vals[2*ind]+learned_vals[2*ind+1], 0, 1)
        trial_results = [y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, nums_ordering]
        all_results.append(trial_results)

    pickle.dump([args, all_results], open('interpret_results/results/evalrank_alg{0}_d{1}_v{2}.pkl'.format(args.fairness_alg, args.dataset, args.version), 'wb'))
    print("Took {:.4f} minutes".format((time.time()-start)/60.))

if __name__ == '__main__':
    main()



