import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import json
from nngallery import MixtureDensityNetwork,BasicClassifier,BasicRegressor
from sklearn.svm import SVR,SVC
from sklearn.tree import DecisionTreeRegressor
from torch import nn
from sklearn.linear_model import LinearRegression,BayesianRidge,Ridge
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

class OrdRegNet(nn.Module):
    def __init__(self, units):
        super(OrdRegNet, self).__init__()
        layers = [nn.BatchNorm1d(units[0])]
        layers=[]
        units = zip(units[:-1], units[1:])
        for in_dim, out_dim in units:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.Dropout(p=0.0,inplace=True))
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


class OrdinalSVM():
    def __init__(self,):
        self.svc=SVC()
        self.anchor=None
        self.fitted=False
    
    def make_data(self,anchor_X,X,anchor_y=None,test_y=None):
        N = len(anchor_X)
        M = len(X)
        pairwise_features = np.concatenate([
            np.repeat(np.reshape(X, [M, 1, -1]), N, axis=1),
            np.repeat(np.reshape(anchor_X, [1, N, -1]), M, axis=0),
        ], axis=-1)
        pairwise_features = np.reshape(pairwise_features, [N*M, -1])

        if anchor_y is not None and test_y is not None:
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
        pred=self.svc.predict(X)
        pred=np.reshape(pred,(M,-1)).sum(axis=-1).astype('int')
        return sorted(anchor_y[pred])





 
if __name__ =="__main__":
    matplotlib.use('AGG')
    parser = ArgumentParser()
    parser.add_argument('data')
    args = parser.parse_args()
    
    pathname,ext=os.path.splitext(args.data)
    path,data_name=os.path.split(pathname)
    # if ext=='.zip':
    #     z=zipfile.ZipFile(args.data)
    #     print(z.printdir())
    #     guessed_name=f'{name}.data'
    #     if guessed_name not in z.namelist():
    #         raise FileNotFoundError
    #     f=z.open(guessed_name,'r')
    data=pd.read_csv(args.data,sep=',?\s*',header=None,index_col=None)
    feat=data.iloc[:,:-1].values
    label=data.iloc[:,-1].values
    print(feat)
    print(label)

    

    # data=json.load(open(args.data))
    # feat,label=list(zip(*data))

    # data=pd.read_pickle(args.data)
    # feat=tuple(data['feature'].values)
    # label=tuple(data['mos'].values)

    
    
    
    # feat=np.array(feat)[:,:121]
    arrs=train_test_split(feat,label)
    train_feat,test_feat,train_label,test_label=list(map(np.array,arrs))

    hiddenlayers=nn.Sequential(
        nn.Linear(feat.shape[1],128),
        # nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(128,128),
        # nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(128,128),
        # nn.Dropout(0.2),
        nn.ReLU(),
        # nn.Linear(128,128),
        # nn.ReLU(),
    )

    regressors={
        'KNN':KNeighborsRegressor,
        'LinearRegression':LinearRegression,
        # 'SVR':SVR,
        'DecisionTree':DecisionTreeRegressor,
        'Ridge':Ridge,
        'BayesianRidge':BayesianRidge,
        'LightGBM':lightgbm.LGBMRegressor,
        'XGBoost':xgboost.XGBRegressor, 
        # 'DNNClassifier':lambda:BasicClassifier(output_size=21,hiddenlayers=hiddenlayers,hidden_size=128),
        'DNNRegressor':lambda:BasicRegressor(hiddenlayers=hiddenlayers,hidden_size=128)
        # 'OrdinalSVM':OrdinalSVM
    }

    n_rows=len(regressors)+1
    plt.figure(figsize=(16,5*n_rows),dpi=200)
    for i,(name,regressor) in enumerate(regressors.items()):
        if name in ['DNNClassifier',]:
            new_label=train_label*4
        else:
            new_label=train_label
        regressor=regressor().fit(train_feat,new_label)
        pred=regressor.predict(test_feat)

        if name in ['DNNClassifier',]:
            pred=pred/4

        ax=plt.subplot(n_rows,2,i*2+1,)
        plot_dist(ax,name,train_label,test_label,pred)
        bx=plt.subplot(n_rows,2,i*2+2)
        plot_error(bx,name,pred,test_label)

    # run ordinal regression
    org_config={
        'model_protocol':OrdRegNet,
        'model_args':{
            'units':[feat.shape[1]*2,64,64,64,2]
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
        'log_dir':'exp/log',
        'ignore_check_dirs':False,
        'remove_old':True
    }
    org=OrdinalRegression(**org_config)
    anchor_set=list(zip(train_feat,train_label))
    org.train(anchor_set)
    _,pred=org.eval(anchor_set,list(zip(test_feat,test_label)))

    ax=plt.subplot(n_rows,2,n_rows*2-1)
    plot_dist(ax,'OrdinalRegression',train_label,test_label,pred,rescore=False)
    bx=plt.subplot(n_rows,2,n_rows*2)
    plot_error(bx,'OrdinalRegression',pred,test_label)

    plt.tight_layout()
    plt.savefig(f'/mnt/tmp/{data_name}.png',dpi=200)

    # plt.clf()
    # sns.kdeplot(x=test_label,cumulative=True)
    # sns.kdeplot(x=pred,cumulative=True)
    # plt.title('CDF of objective scores and subjective score')
    # plt.xlabel('Score')
    # plt.legend(['objective','subjective'])
    # plt.savefig('/mnt/tmp/cdf.png',dpi=200)

    # target_dist=GaussianMixture(n_components=6).fit(np.reshape(test_label,(-1,1)))
    # error_dist=GaussianMixture(n_components=1).fit(np.reshape(pred-test_label,(-1,1)))
    # resample_target=target_dist.sample(660)[0].reshape(-1)
    # resample_error=error_dist.sample(660)[0].reshape(-1)
    # fixed_resamples=np.sort(resample_target)[np.argsort(np.argsort(resample_target+resample_error))]
    
    # plt.clf()
    # sns.kdeplot(x=resample_target,y=resample_error,fill=True,cbar=True)
    # plt.title('prediction error distribution')
    # plt.savefig('/mnt/tmp/high_dim_resample_svr_error_dist.png',dpi=200)
