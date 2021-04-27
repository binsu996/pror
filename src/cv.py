import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import json
from nngallery import MixtureDensityNetwork,BasicClassifier,BasicRegressor
from sklearn.svm import SVR,SVC
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from torch import nn
from sklearn.linear_model import LinearRegression,BayesianRidge,Ridge,LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import xgboost
import lightgbm
import pandas as pd
from ordinalregression import OrdinalRegression
import pickle
import pandas as pd
import os
import zipfile
from ordinalregression import EvaluatorEX
from multiprocessing import Pool
from uuid import uuid4
import warnings


class OrdinalSVM():
    def __init__(self,):
        self.svc=LogisticRegression()
        self.anchor=None
        self.fitted=False
    
    def make_data(self,anchor_X,X,anchor_y=None,y=None):
        N = len(anchor_X)
        M = len(X)
        pairwise_features = np.concatenate([
            np.repeat(np.reshape(X, [M, 1, -1]), N, axis=1),
            np.repeat(np.reshape(anchor_X, [1, N, -1]), M, axis=0),
        ], axis=-1)
        pairwise_features = np.reshape(pairwise_features, [N*M, -1])

        if anchor_y is not None and y is not None:
            pairwise_labels = np.greater_equal(
                np.reshape(y, [M, 1, -1]),
                np.reshape(anchor_y, [1, N, -1])
            ).astype('int')
            pairwise_labels = np.reshape(pairwise_labels, -1)
            return pairwise_features,pairwise_labels
        
        return pairwise_features
        
    
    def fit(self,X,y):
        self.anchor=X,y
        X,y=self.make_data(X,X,y,y)
        self.svc.fit(X,y)
        self.fitted=True
    
    def predict(self,X):
        M=len(X)
        anchor_X,anchor_y=self.anchor
        X=self.make_data(anchor_X,X)
        pred=self.svc.predict_proba(X)[:,1]
        pred=np.reshape(pred,(M,-1)).sum(axis=-1).astype('int')
        return sorted(anchor_y[pred])

class OrdRegNet(nn.Module):
    def __init__(self, units):
        super(OrdRegNet, self).__init__()
        layers = [nn.BatchNorm1d(units[0])]
        layers=[]
        units = zip(units[:-1], units[1:])
        for in_dim, out_dim in units:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Dropout(p=0.5,inplace=True))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers[:-3]).cuda()

    def forward(self, x):
        return self.net(x)

def plot_dist(ax,name,train_label,test_label,pred,rescore=True):
    
    sns.kdeplot(train_label,ax=ax,cumulative=True)
    sns.kdeplot(test_label,ax=ax,cumulative=True)
    sns.kdeplot(pred,ax=ax,cumulative=True)
    legend=[
        'train',
        'test',
        f"pred (MAE:{np.mean(np.abs(pred-test_label)):.2f},PCC:{np.corrcoef(pred,test_label)[0][1]:.2f})"
    ]

    if rescore:
        anchor=train_label
        order=np.argsort(np.argsort(pred))
        n,m=len(anchor),len(test_label)
        fix=np.sort(anchor)[list(map(lambda x:x*n//m,order))]
        sns.kdeplot(fix,ax=ax,cumulative=True)
        legend.append(f'rescored (MAE:{np.mean(np.abs(fix-test_label)):.2f},PCC:{np.corrcoef(fix,test_label)[0][1]:.2f})')

    plt.xlabel('Score')
    plt.legend(legend)
    plt.title(f'{name}: Data distributions')

def plot_error(ax,name,pred,test_label):
    sns.kdeplot(x=test_label,y=pred-test_label,fill=True,cbar=True,ax=ax)
    plt.xlabel('Score')
    plt.ylabel('Diff')
    plt.title(f'{name}: Prediction error distribution')


def load_data(train_filename,test_filename):
    train=pd.read_csv(train_filename,sep=',?\s*',header=None,index_col=None)
    test=pd.read_csv(test_filename,sep=',?\s*',header=None,index_col=None)
    train_X=train.iloc[:,:-1].values
    train_y=train.iloc[:,-1].values
    test_X=test.iloc[:,:-1].values
    test_y=test.iloc[:,-1].values
    return train_X,train_y,test_X,test_y

def nnregression(train_X,train_y,test_X,test_y):
    train_X,valid_X,train_y,valid_y=train_test_split(train_X,train_y,random_state=0,shuffle=True)
    hiddenlayers=nn.Sequential(
        nn.Linear(train_X.shape[1],128),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(128,128),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(128,128),
        nn.Dropout(0.5),
        nn.ReLU(),
    )
    regressor=BasicRegressor(hiddenlayers=hiddenlayers,hidden_size=128)
    regressor.fit(train_X,train_y,valid_X,valid_y,lr=1e-3,epoch=256)
    pred_y=regressor.predict(test_X)
    evalutor=EvaluatorEX()
    return evalutor.eval(pred_y,test_y,1)

def osvm(train_X,train_y,test_X,test_y):
    # train_X,valid_X,train_y,valid_y=train_test_split(train_X,train_y,random_state=0,shuffle=True)
    regressor=DecisionTreeRegressor()
    regressor.fit(train_X,train_y)
    pred_y=regressor.predict(test_X)
    evalutor=EvaluatorEX()
    return evalutor.eval(pred_y,test_y,1)
    
def ORARS(train_X,train_y,test_X,test_y):
    train_X,valid_X,train_y,valid_y=train_test_split(train_X,train_y,random_state=0,shuffle=True)
    org_config={
        'model_protocol':OrdRegNet,
        'model_args':{
            'units':[train_X.shape[1]*2,128,128,128,2]
        },
        'config':{
            'lr':1e-4,
            'cuda':True,
            'generate_weight':True,
            'bin_width':3,
            'batch_size':1024,
            'soft_scale':0.5,
            'epoch':4,
        },
        'log_dir':f'exp/{uuid4()}',
        'ignore_check_dirs':False,
        'remove_old':True
    }
    org=OrdinalRegression(**org_config)
    anchor_set=list(zip(train_X,train_y))
    valid_set=list(zip(valid_X,valid_y))
    org.train(anchor_set,valid_set,show_time=False)
    return org.eval(anchor_set,list(zip(test_X,test_y)),load_best_model=True)[0]


 
if __name__ =="__main__":
    matplotlib.use('AGG')
    parser = ArgumentParser()
    parser.add_argument('prefix')
    parser.add_argument('bin')
    args = parser.parse_args()

    name=args.prefix.split('/')[-2]
    bias=int(os.path.exists(f'{args.prefix}/train_{name}.{20}'))
    gen_trial_params=lambda i:[f'{args.prefix}/train_{name}.{i}',f'{args.prefix}/test_{name}.{i}']
    trial_params=map(gen_trial_params,range(0+bias,20+bias))
    trial_data=list(map(load_data,*zip(*trial_params)))
    pool=Pool(8)

    res=pool.starmap(nnregression,trial_data)
    evaluator=EvaluatorEX()
    for x in res:
        evaluator.add_record(x)
    print('NNR',evaluator.get_current_mean())

    # res=pool.starmap(osvm,trial_data)
    # evaluator=EvaluatorEX()
    # for x in res:
    #     evaluator.add_record(x)
    # print('OrdinalSVM',evaluator.get_current_mean())

    res=[ORARS(*x) for x in trial_data]
    # res=pool.starmap(ORARS,trial_data)
    evaluator=EvaluatorEX()
    for x in res:
        evaluator.add_record(x)
    print('ORARS',evaluator.get_current_mean())


