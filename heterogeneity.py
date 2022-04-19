import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
import copy
from sklearn.neighbors import KNeighborsClassifier
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

    datasets = [ACSIncomeDem, ACSPublicCoverageDem, ACSMobilityDem, ACSEmploymentFilteredDem, ACSTravelTimeDem]
    dataset = datasets[args.dataset]

    features, label, group = dataset.df_to_numpy(acs_data)

    cont_feats = ['AGEP', 'SCHL', 'WKHP', 'PINCP', 'JWMNP', 'POVPIP'] # could also be discretized
    cat_feats = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'ESR', 'ST', 'FER', 'GCL', 'POWPUMA', 'JWTR', 'PUMA', 'RAC1P', 'RAC2P', 'RAC3P']


    num_granular = 0
    remove_indices = []
    race_names = []
    remove_names = []

    if args.group == 0:
        group_consider = [5, 6]
    elif args.group == 1:
        group_consider = [2, 3]
    else:
        group_consider = []

    for rac1p in range(len(rac1p_names)):
        these_indices = np.where(features[:, -3] == rac1p+1)[0]
        these_features, these_labels = features[these_indices], label[these_indices]
        if args.bwmf:
            if rac1p in [0, 1]:
                race_names.append(rac1p_names[rac1p] + ' Female')
                race_names.append(rac1p_names[rac1p] + ' Male')
            else:
                remove_indices.append(these_indices)
                remove_names.append(rac1p_names[rac1p])
            continue

        if rac1p in group_consider:
            if args.tier == 0 and rac1p in [3, 6]:
                if rac1p == 3:
                    race_names.append('AI_AN')
                if rac1p == 6:
                    race_names.append('A_NHOPI')
            if args.tier == 1:
                race_names.append(rac1p_names[rac1p])

            if rac1p == 6 and len(states) == 1 and states[0] == 'CA': # hard-code that makes PI stay one group instead of granular for CA
                if len(these_indices) > min_samples:
                    if np.sum(these_labels) > min_positives and np.sum(these_labels==0) > min_negatives:
                        if args.tier == 2:
                            race_names.append(rac1p_names[rac1p])
                        continue
                remove_indices.append(these_indices)
                remove_names.append(rac1p_names[rac1p])
                continue

            races, counts = np.unique(these_features[:, -2], return_counts=True)
            for r in range(len(races)):
                this_race = np.where(these_features[:, -2]==races[r])[0]
                if counts[r] > min_samples:
                    if np.sum(these_labels[this_race]) > min_positives and np.sum(these_labels[this_race]==0) > min_negatives:
                        if args.tier == 2:
                            race_names.append('{0}-{1}'.format(rac1p_names[rac1p], races[r]))
                        num_granular += 1
                        continue
                remove_indices.append(these_indices[this_race])
                remove_names.append('{0}-{1}'.format(rac1p_names[rac1p], races[r]))
        else:
            if args.hetero_bw_api:
                if rac1p_names[rac1p] not in ['White', 'Black']:
                    remove_indices.append(these_indices)
                    remove_names.append(rac1p_names[rac1p])
                    continue
            if len(these_indices) > min_samples:
                if np.sum(these_labels) > min_positives and np.sum(these_labels==0) > min_negatives:
                    race_names.append(rac1p_names[rac1p])
                    continue
            remove_indices.append(these_indices)
            remove_names.append(rac1p_names[rac1p])

    remove_indices = np.concatenate(remove_indices)
    keep_indices = np.array(list(set(np.arange(len(features))).difference(set(remove_indices))))
    features, label = features[keep_indices], label[keep_indices]

    nums_ordering = [race_names]

    features = pd.DataFrame(features, columns=dataset.features)

    if args.other_exp > 0:
        to_del_feats = []
        for feat in dataset.features:
            if 'RAC2P' in feat or 'RAC3P' in feat:
                to_del_feats.append(feat)
        features = features.drop(to_del_feats,axis=1)

        if args.group == 2:
            other_group_idx = 8
        elif args.group == 3:
            other_group_idx = 9
        race_feats = ['RAC1P']
    else:
        race_feats = ['RAC1P', 'RAC2P', 'RAC3P']

    these_cat_feats = []
    for feat in features.keys():
        if feat in cat_feats:
            these_cat_feats.append(feat)

    no_race_feats = list(set(these_cat_feats).difference(set(race_feats)))
    data_cat = features[these_cat_feats]
    data_cat = pd.get_dummies(data_cat.astype(int).astype(str), drop_first=False)
    features = pd.concat([features.drop(no_race_feats,axis=1), data_cat], axis=1)

    if args.other_exp == 0 and not args.bwmf and args.group == 0:
        for key in features.keys():
            if 'RAC' in key:
                race_feats.append(key)

    if args.protatt in [1, 2]:
        for key in features.keys():
            if 'SEX' in key or 'RAC' in key:
                race_feats.append(key)

    if args.protatt == 2:
        assert args.bwmf # currently hard-coded for this case
        for r in [0, 1]:
            for s in [0, 1]:
                features['R-{0}_S-{1}'.format(rac1p_names[r], 'Female' if s == 0 else 'Male')] = (np.array(features['RAC1P'])==r+1)*features['SEX_{}'.format(s+1)]

    features = features.loc[:,~features.columns.duplicated()]

    return features, label, rac1p_names, remove_names, states, race_feats, nums_ordering

def split_dataset(args, features, label, rac1p_names, remove_names, states, race_feats, nums_ordering):
    mid_X_train, X_test, mid_y_train, y_test = train_test_split(features, label, test_size=.3)
    X_train, X_val, y_train, y_val = train_test_split(mid_X_train, mid_y_train, test_size=.3)

    if args.other_exp > 0:
        if args.group == 2:
            other_group_idx = 8
        elif args.group == 3:
            other_group_idx = 9

    if args.bwmf and args.bf_smalln > -1 and args.bf_train == -1:
        # change the training number of bf
        bf_indices = set(np.where(np.array(X_train['SEX_2'])==1)[0])&set(np.where(np.array(X_train['RAC1P']) == 2)[0])
        sample_size = min(args.bf_smalln, len(bf_indices))
        remove = np.random.choice(list(bf_indices), size=len(bf_indices)-sample_size, replace=False)
        keep = np.array(list(set(np.arange(len(X_train))).difference(set(remove))))
        X_train, y_train = X_train.iloc[keep], y_train[keep]

    if args.other_exp == 2:
        # Other is 8, Two or more is 9
        these_indices = np.where(np.array(X_train['RAC1P']) == other_group_idx)[0]
        keep_indices = np.array(list(set(np.arange(len(X_train))).difference(set(these_indices))))
        X_train, y_train = X_train.iloc[keep_indices], y_train[keep_indices]
    elif args.other_exp == 1:
        
        group_start = time.time()
        to_del_feats = []
        for key in features.keys():
            if 'RAC' in key:
                to_del_feats.append(key)
        group_features = X_train.drop(to_del_feats,axis=1)
        group_label = np.array(X_train['RAC1P'])
        drop_indices = np.where(group_label==other_group_idx)[0]
        group_features, group_label = np.delete(np.array(group_features), drop_indices, axis=0), np.delete(group_label, drop_indices, axis=0)
        group_model = KNeighborsClassifier(n_neighbors=1)

        group_X_train, group_X_val, group_y_train, group_y_val = train_test_split(group_features, group_label, test_size=.3)
        group_model.fit(group_X_train, group_y_train)
        preds = group_model.predict(group_X_val)
        group_end = time.time()

        races, counts = np.unique(group_y_train, return_counts=True)

        other_samples = X_train.iloc[drop_indices]
        other_samples = other_samples.drop(to_del_feats, axis=1)
        preds_other = group_model.predict(other_samples)

        actual_rac1ps = []
        for x, X in enumerate([X_train, X_val, X_test]):
            actual_group = np.array(X['RAC1P'])

            predict_indicator = (np.array(X['RAC1P']) == other_group_idx)
            predict_indices = np.where(predict_indicator)[0]
            pred_features = X.drop(to_del_feats, axis=1).iloc[predict_indices]
            pred_group = group_model.predict(pred_features)

            new_values = np.array(X['RAC1P'])
            new_values[predict_indices] = pred_group
            X.loc[:, 'RAC1P'] = new_values
            new_values = np.array(X['RAC1P_{}'.format(other_group_idx)])
            new_values[predict_indices] = 0
            X.loc[:, 'RAC1P_{}'.format(other_group_idx)] = new_values

            for p in np.unique(pred_group):
                these_p_features = np.array(X['RAC1P_{}'.format(int(p))])
                indices_within = np.where(pred_group==p)[0]
                these_p_features[predict_indices[indices_within]] = 1
                X.loc[:, 'RAC1P_{}'.format(int(p))] = these_p_features

            if x == 0:
                X_train = copy.deepcopy(X)
            elif x == 1:
                X_val = copy.deepcopy(X)
            elif x == 2:
                X_test = copy.deepcopy(X)

            actual_rac1ps.append(actual_group)

    if args.group == 0:
        group_consider = [5, 6]
    elif args.group == 1:
        group_consider = [2, 3]
    else:
        group_consider = []


    labels_train = []
    all_indices = []
    granular_indices = []
    mid_indices = []

    blas = []
    for x, X in enumerate([X_train, X_val, X_test]):
        this_indices = []
        this_granular_indices = []
        this_mid_indices = []

        for rac1p in range(len(rac1p_names)):
            if args.other_exp == 0:
                assert len(X['RAC2P'].shape)==1
            these_indices = np.where(np.array(X['RAC1P']) == rac1p+1)[0]
            these_features = X.iloc[these_indices]


            if rac1p_names[rac1p] not in remove_names:
                if args.other_exp == 1:
                    temp_indices = np.where(actual_rac1ps[x]==rac1p+1)[0]
                    this_mid_indices.append(temp_indices)
                else:
                    this_mid_indices.append(these_indices)

            if args.bwmf:
                if rac1p in [0, 1]:
                    male_indices = np.where(np.array(X['SEX_1'])==1)[0]
                    female_indices = np.where(np.array(X['SEX_2'])==1)[0]
                    race_male_indices = np.array(list(set(these_indices)&set(male_indices)))
                    race_female_indices = np.array(list(set(these_indices)&set(female_indices)))
                    this_indices.append(race_female_indices)
                    this_granular_indices.append(race_female_indices)
                    this_indices.append(race_male_indices)
                    this_granular_indices.append(race_male_indices)

                    if x == 0:
                        indicator = np.zeros(len(X))
                        indicator[race_female_indices] = 1
                        labels_train.append(indicator)
                        indicator = np.zeros(len(X))
                        indicator[race_male_indices] = 1
                        labels_train.append(indicator)
                continue

            if args.other_exp == 0 and ((len(states) == 1 and states[0] == 'CA' and rac1p in [5]) or (states[0] != 'CA' and len(states) != 1 and rac1p in group_consider)): # hard-code that makes PI stay one group instead of granular for CA. i.e., for CA we only enter this loop for Asian
                races, counts = np.unique(np.array(these_features['RAC2P']), return_counts=True)

                if args.tier == 0:
                    if rac1p in [2, 5]:
                        these_indices_after = np.where(np.array(X['RAC1P']) == rac1p+2)[0]
                        these_indices_concat = np.concatenate([these_indices_after, these_indices])
                        this_indices.append(these_indices_concat)
                        if x == 0:
                            indicator = np.zeros(len(X))
                            indicator[these_indices_concat] = 1
                            labels_train.append(indicator)
                if args.tier == 1:
                    this_indices.append(these_indices)
                    if x == 0:
                        indicator = np.zeros(len(X))
                        indicator[these_indices] = 1
                        labels_train.append(indicator)
                if args.tier == 2:
                    for r in range(len(races)):
                        name = '{0}-{1}'.format(rac1p_names[rac1p], races[r])
                        if name in remove_names:
                            continue
                        this_race = np.where(np.array(these_features['RAC2P'])==races[r])[0]
                        this_indices.append(these_indices[this_race])
                        if x == 0:
                            indicator = np.zeros(len(X))
                            indicator[these_indices[this_race]] = 1
                            labels_train.append(indicator)

                for r in range(len(races)):
                    name = '{0}-{1}'.format(rac1p_names[rac1p], races[r])
                    if name in remove_names:
                        continue
                    this_race = np.where(np.array(these_features['RAC2P'])==races[r])[0]
                    this_granular_indices.append(these_indices[this_race])
            else:
                if args.tier == 0 and rac1p == 6:
                    this_granular_indices.append(these_indices)
                    continue
                if rac1p_names[rac1p] in remove_names:
                    continue
                this_indices.append(these_indices)
                this_granular_indices.append(these_indices)
                if x == 0:
                    indicator = np.zeros(len(X))
                    indicator[these_indices] = 1
                    labels_train.append(indicator)
        all_indices.append(this_indices)
        granular_indices.append(this_granular_indices)
        mid_indices.append(this_mid_indices)

    if args.other_exp == 0 and not args.bwmf and args.group == 0: 
        for x, X in enumerate([X_train, X_val, X_test]):
            for ind in range(len(all_indices[0])):
                replace_features = np.zeros(len(X))
                replace_features[all_indices[x][ind]] = 1
                if x == 0:
                    X_train.loc[:, 'API_Hetero_R{}'.format(ind)] = replace_features
                elif x == 1:
                    X_val.loc[:, 'API_Hetero_R{}'.format(ind)] = replace_features
                elif x == 2:
                    X_test.loc[:, 'API_Hetero_R{}'.format(ind)] = replace_features

    min_max_scaler = preprocessing.MinMaxScaler()
    for x, X in enumerate([X_train, X_val, X_test]):
        X = X.drop(race_feats,axis=1)
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
    assert len(labels_train) == len(all_indices[0])
    assert np.sum(labels_train[2]) == len(all_indices[0][2])
    if len(labels_train) > 6:
        assert np.sum(labels_train[6]) == len(all_indices[0][6])
    if args.tier == 2:
        assert len(granular_indices[0][3]) == len(all_indices[0][3])
        assert np.sum(granular_indices[0][3]) == np.sum(all_indices[0][3])
        if len(labels_train) > 5:
            assert np.sum(granular_indices[1][5]) == np.sum(all_indices[1][5])
        if len(labels_train) > 6:
            assert np.sum(granular_indices[2][6]) == np.sum(all_indices[2][6])
    if args.tier == 1:
        if args.other_exp in [0, 3]:
            assert len(mid_indices[0][3]) == len(all_indices[0][3])
            assert np.sum(mid_indices[0][3]) == np.sum(all_indices[0][3])
            if len(labels_train) > 5:
                assert np.sum(mid_indices[1][5]) == np.sum(all_indices[1][5])
            if len(labels_train) > 6:
                assert np.sum(mid_indices[2][6]) == np.sum(all_indices[2][6])


    if args.bf_train > -1:
        assert args.bwmf

        if args.bf_train < 4:
            min_length = np.amin([len(all_indices[0][g]) for g in range(len(all_indices[0]))])
            keep_indices = np.random.choice(all_indices[0][args.bf_train], min_length, replace=True)
            assert len(keep_indices)==min_length
            X_train, y_train = X_train[keep_indices], y_train[keep_indices]
            for g in range(len(all_indices[0])):
                if g == args.bf_train:
                    all_indices[0][g] = np.arange(len(X_train))
                    indicator = np.ones(len(X_train))
                else:
                    all_indices[0][g] = []
                    indicator = np.zeros(len(X_train))
                labels_train[g] = indicator
        else:
            if args.bf_train in np.arange(4, 15):
                focus_idx = 2
                minus_num = 4
            elif args.bf_train in np.arange(15, 26):
                focus_idx = 1
                minus_num = 15

            # 2 is bf, 0 is wf, and 1 is wm, 3 is bm
            n_bf = len(all_indices[0][focus_idx])
            ratios = np.arange(0, 1.1, .1)
            # 4-14 are all different combos of just bm+wf
            bf_smalln = args.bf_smalln
            if args.bf_smalln == -1:
                bf_smalln = n_bf
            bf_indices = np.random.choice(all_indices[0][focus_idx], bf_smalln)
            remainder = n_bf-bf_smalln

            bm_indices = np.random.choice(all_indices[0][3], int(remainder*ratios[args.bf_train-minus_num]))
            wf_indices = np.random.choice(all_indices[0][0], int(remainder*(1.-ratios[args.bf_train-minus_num])))
            train_indices = np.concatenate([bf_indices, bm_indices, wf_indices])
            X_train, y_train = X_train[train_indices], y_train[train_indices]
            all_indices[0][focus_idx] = bf_indices
            all_indices[0][0] = wf_indices
            all_indices[0][3] = bm_indices
            all_indices[0][3-focus_idx] = []
            for g in range(len(all_indices[0])):
                indicator = np.zeros(len(X_train))
                if g == 0:
                    indicator[len(bf_indices)+len(bm_indices):] = 1
                elif g == focus_idx:
                    indicator[:len(bf_indices)] = 1
                elif g == 3:
                    indicator[len(bf_indices):len(bf_indices)+len(bm_indices)] = 1
                labels_train[g] = indicator


    return X_train, y_train, X_val, y_val, X_test, y_test, all_indices, mid_indices, granular_indices, nums_ordering, labels_train

def main():
    parser = argparse.ArgumentParser(description='intersectionality')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--num_trials', type=int, default=5, help='number of trials')
    parser.add_argument('--dataset', type=int, default=0, help='which of 5 folktable datasets from 2018')
    parser.add_argument('--group', type=int, default=0, help='0 is API, 1 is AI+AN, 2 is other and 3 is biracial (only applicable for other_exp)')
    parser.add_argument('--tier', type=int, default=0, help='0 is args.group is all together, 1 is 2 groups, 2 is all disaggregated')
    parser.add_argument('--other_exp', type=int, default=0, help='0 is regular setting, so other is their own thing, 1 is you try to distribute each of args.group into a different group, and 2 is remove all other, 3 is also regular setting but you just set this to know and it drops rac2p rac3p')
    parser.add_argument('--protatt', type=int, default=0, help='0 is protatt a feature, 1 is no protected attributes, 2 is protatt only as race-sex')
    parser.add_argument('--bwmf', action='store_true', default=False, help='if set, only [Black, white] x [female, male] are kept. This also means that now gender is considered')
    parser.add_argument('--bf_train', type=int, default=-1, help='if bwmf is set, but this impacts which group is used as just training, so 0, 1 is something, 2 is bf, 3 is something else. -1 is everything. 4-14 is a bunch of ratios of bm+wf, with however many bf that bf_smalln specifies')
    parser.add_argument('--hetero_bw_api', action='store_true', default=False, help='if set, for the hetero api experiment, all non-Black/white/API are discarded')
    parser.add_argument('--bf_smalln', type=int, default=-1, help='the number of bw, if -1 then just default number')
    parser.add_argument('--fairness_alg', type=int, default=0, help='all of the different methods we have. if -1 then does not train, makes dataset for yang')
    parser.add_argument('--warnings', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    start = time.time()

    # to make sure command line arguments make sense

    if args.other_exp == 0 and not args.bwmf:  # the API heterogeneity experiment
        assert args.group in [0, 1]
        assert args.bf_smalln == -1 # default
        assert args.bf_train == -1 # default
        # can set args.hetero_bw_api
        # set args.tier probably

    if args.bwmf: # (1) small-n for BF to test the train vs test accuracy for overfitting on this group, (2) making the 5x4 to see if trained on one group and tested on other
        assert args.group == 0 # aka default and doesn't matter
        assert args.tier == 0 # aka default and doesn't matter
        assert args.other_exp == 0 # aka default and doesn't matter

        if args.bf_smalln > -1 and args.bf_train == -1: # if doing (1)
            assert args.protatt == 0 # default
            # set args.bf_smalln 

        if args.bf_train > -1: # if doing (2)
            assert args.protatt == 0 # default
            # set args.bf_train
            # can also set args.bf_smalln for a special case

    if args.other_exp > 0: # the Other racial group experiment
        assert args.group in [2, 3]
        assert args.tier == 1
        assert args.bf_smalln == -1 # default
        assert args.bf_train == -1 # default
        assert not args.bwmf # aka default and doesn't matter
        # set args.other_exp probably

    if args.seed == -1:
        args.seed = np.random.randint(0, 1000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print("Random seed: {}".format(args.seed))

    features, label, rac1p_names, remove_names, states, race_feats, nums_ordering = create_dataset(args)

    all_results = []

    for i in range(args.num_trials):

        X_train, y_train, X_val, y_val, X_test, y_test, all_indices, mid_indices, granular_indices, nums_ordering, labels_train = split_dataset(args, features, label, rac1p_names, remove_names, states, race_feats, nums_ordering)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if args.fairness_alg in [14, 15, 16, 17, 18, 19, 29, 30, 31, 32, 33, 34]:
            if args.fairness_alg in [14, 15, 16, 17, 18, 19]:
                probs_train, probs_val, probs_test = train_model_rdc(args, X_train, y_train, labels_train, nums_ordering, device, all_indices, X_val, X_test)
            elif args.fairness_alg in [29, 30, 31, 32, 33, 34]:
                probs_train, probs_val, probs_test = train_model_gry(args, X_train, y_train, labels_train, nums_ordering, device, all_indices, X_val, X_test)
            else:
                raise NotImplementedError
        else:
            model, all_models, postproc_eps = train_model(args, X_train, y_train, labels_train, nums_ordering, device, att_feats=None, curr_indices=all_indices, X_val=None, y_val=None, name='{0}-{1}-{2}-{3}'.format(args.fairness_alg, i, args.dataset, args.bf_smalln))

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

        trial_results = [y_train, y_val, y_test, probs_train, probs_val, probs_test, all_indices, mid_indices, granular_indices, nums_ordering]
        all_results.append(trial_results)

    if args.bwmf:
        pickle.dump([args, all_results], open('interpret_results/results/alg{0}_d{1}_p{2}_n{3}_bf{4}.pkl'.format(args.fairness_alg, args.dataset, args.protatt, args.bf_smalln, args.bf_train), 'wb'))
    elif args.other_exp > 0:
        pickle.dump([args, all_results], open('interpret_results/results/alg{0}_d{1}_g{2}_o{3}.pkl'.format(args.fairness_alg, args.dataset, args.group, args.other_exp), 'wb'))
    else:
        pickle.dump([args, all_results], open('interpret_results/results/alg{0}_d{1}_g{2}_t{3}_bw{4}.pkl'.format(args.fairness_alg, args.dataset, args.group, args.tier, '1' if args.hetero_bw_api else '0'), 'wb'))
    print("Took {:.4f} minutes".format((time.time()-start)/60.))

if __name__ == '__main__':
    main()



