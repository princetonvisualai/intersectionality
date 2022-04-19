import fairlearn.reductions as reductions
import numpy as np
import sklearn.neural_network as sknn
import sklearn.svm as svm
import sklearn.linear_model as lm
import torch
import pickle
import torch
import torch.nn as nn
import os

class NeuralNet:
    def __init__(self, input_size, device, max_iter, batch_size):
        self.ann = nn.Sequential(
            nn.Linear(input_size, 30), 
            nn.ReLU(), 
            nn.Linear(30, 30), 
            nn.ReLU(), 
            nn.Linear(30, 1) 
            )
        self. optimizer = torch.optim.SGD(self.ann.parameters(), lr=0.005, momentum=0.9)
        self.device = device
        self.dtype = torch.float32
        self.COUNT = 0        
        self.max_iter = max_iter 
        self.batch_size = batch_size
    
    def forward(self, x):
        features = self.ann(x)
        return features
   
    
    def fit(self, X, y, sample_weight=None):
        batch_size=self.batch_size
        N = X.shape[0]//batch_size +1
        self.COUNT+=1
        X = np.array(X)
        y = np.array(y)
        sample_weight = np.array(sample_weight)

        self.ann.to(self.device)
        self.ann.train()
        for e in range(self.max_iter):
            perm = list(np.random.permutation(X.shape[0]))
            X = X[perm]
            y = y[perm]
            if sample_weight is not None:
                sample_weight=sample_weight[perm]
            for t in range(N):
                
                x_batch = torch.Tensor(X[t*batch_size:(t+1)*batch_size]).to(self.device)
                y_batch = torch.Tensor(y[t*batch_size:(t+1)*batch_size]).to(self.device)
                sample_weight_batch = torch.Tensor(sample_weight[t*batch_size:(t+1)*batch_size]).to(self.device)
                scores = self.ann(x_batch)
                self.optimizer.zero_grad()
                lossbce = torch.nn.BCEWithLogitsLoss(weight=sample_weight_batch)
                
                loss = lossbce(scores.reshape(-1), y_batch.reshape(-1))
                    
                loss.backward()
                self.optimizer.step()

    def predict(self, X):

        self.ann.eval()
        with torch.no_grad():
            pred = self.ann(torch.Tensor(X).to(self.device))
            pred = torch.sigmoid(pred).squeeze()
        
        return pred.cpu().numpy()

    def save(self, path):
        torch.save({'ann':self.ann.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)
        

def train_model(args, X_train, y_train, labels_train, nums_ordering, device, 
                all_indices, X_val, X_test):
    
    if args.fairness_alg==14:
        batch_size, max_iter_nn, max_iter_overall = 512, 50, 10
    elif args.fairness_alg==15:
        batch_size, max_iter_nn, max_iter_overall = 512, 50, 50
    elif args.fairness_alg==16:
        batch_size, max_iter_nn, max_iter_overall = 1024, 200, 20
    elif args.fairness_alg==17:
        batch_size, max_iter_nn, max_iter_overall = 2048, 100, 50
    elif args.fairness_alg==18:
        batch_size, max_iter_nn, max_iter_overall = 256, 50, 10
    elif args.fairness_alg==19:
        batch_size, max_iter_nn, max_iter_overall = 1024, 100, 20

    prot_train = np.zeros((X_train.shape[0], len(labels_train)))
    for i, lab in enumerate(labels_train):
       prot_train[:, i] = lab
    

    constraints  = reductions.TruePositiveRateParity()
    classifier = NeuralNet(X_train.shape[1], device, max_iter_nn, batch_size)
    ExpGrad = reductions.ExponentiatedGradient(classifier, constraints, max_iter = max_iter_overall)

    kwargs = {'sensitive_features': prot_train}
    ExpGrad.fit(X_train, y_train, **kwargs)
        

    preds = {}

    for name, split in [('train', X_train), ('val', X_val), ('test', X_test)]:
        this_pred = np.zeros(split.shape[0])

        for i in range(ExpGrad.weights_.shape[0]):
            this_pred += ExpGrad.weights_[i]* ExpGrad.predictors_[i].predict(split)
            #if ExpGrad.weights_[i] >0:
            #    #ExpGrad.predictors_[i].save('{}/pred{}.pth'.format(save_folder, i))
        preds[name] = this_pred
        

    soft_acc = (preds['train']*y_train+(1-preds['train'])*(1-y_train)).mean()
    return preds['train'], preds['val'], preds['test']

