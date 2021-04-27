import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import json
from nngallery import MixtureDensityNetwork,BasicClassifier,BasicRegressor
from sklearn.svm import SVR,SVC
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from torch import nn
import pandas as pd
from ordinalregression import OrdinalRegression,CrossValidation
import pickle
import pandas as pd
import os
from ordinalregression import EvaluatorEX
from uuid import uuid4
import warnings
from ray import tune
import ray
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

# def process_data(filename,nfold):
#     dataset=pd.read_csv(filename,sep=',\s*|\s+',header=None,index_col=None)
#     cv=CrossValidation(dataset.values,nfold,shuffle_seed=666,split_seed=888)
#     data=[x for x in cv]
#     dir_path=filename+'_cv'
#     os.makedirs(dir_path,exist_ok=True)
#     cv_data=[]
#     for i in range(len(data)):
#         dataset.to_csv(os.path.join(dir_path,'data'))
#         train,valid,test=data[i]
#         pd.DataFrame(data=train).to_csv(os.path.join(dir_path,f'train.{i}'),header=False,index=False)
#         pd.DataFrame(data=valid).to_csv(os.path.join(dir_path,f'valid.{i}'),header=False,index=False)
#         pd.DataFrame(data=test).to_csv(os.path.join(dir_path,f'test.{i}'),header=False,index=False)
#         train_valid_test=[train[:,:-1],train[:,-1],valid[:,:-1],valid[:,-1],test[:,:-1],test[:,-1]]
#         pickle.dump(train_valid_test,open(os.path.join(dir_path,f'cv.{i}'),'wb'))
#         cv_data.append(train_valid_test)
#     pickle.dump(cv_data,open(os.path.join(dir_path,f'cv'),'wb'))
#     print('Data processing is OK!')

def process_data(filename,nfold,name):
    dataset=pd.read_csv(filename,sep=',\s*|\s+',header=None,index_col=None)
    dataset=list(range(len(dataset)))
    cv=CrossValidation(dataset,nfold,shuffle_seed=666,split_seed=888)
    data=[x for x in cv]
    cv_data=[]
    for i in range(len(data)):
        train,valid,test=data[i]
        cv_data.append({
            i:{
                "train_indexs":train.astype('int').tolist(),
                "valid_indexs":valid.astype('int').tolist(),
                "test_indexs":test.astype('int').tolist()
            }
        })
    json.dump(cv_data,open(f"tmp/{name}.json",'w'))
    print('Data processing is OK!')
    exit()

def load_data(path):
    return pickle.load(open(path,'rb'))

def search_nnr(config):
    cv_data=load_data(config['data'])
    unit=config['unit']
    dropout=config['dropout']
    evalutor=EvaluatorEX()
    val_losses=[]
    test_maes=[]
    rescore_maes=[]
    metas=[]
    for train_X,train_y,valid_X,valid_y,test_X,test_y in cv_data:
        hiddenlayers=nn.Sequential(
            nn.Linear(train_X.shape[1],unit),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(unit,unit),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(unit,unit),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        regressor=BasicRegressor(hiddenlayers=hiddenlayers,hidden_size=unit)
        regressor.fit(train_X,train_y,valid_X,valid_y,lr=config['lr'],epoch=config['epoch'],cuda=False,batch_size=32)
        val_losses.append(regressor.val_loss)
        pred_y=regressor.predict(test_X,cuda=False)
        test_maes.append(np.mean(np.abs(pred_y-test_y)))

        rescored=regressor.predict(np.concatenate([train_X,valid_X],axis=0))
        N=len(rescored)
        idxs=np.sum((pred_y.reshape(-1,1)>rescored).astype('int'),axis=1).clip(max=N-1)
        gts=np.concatenate([train_y,valid_y]).reshape(-1)
        gts=np.sort(gts)
        rty=gts[idxs]
        rescore_maes.append(np.mean(np.abs(rty-test_y)))
        metas.append({
            'gt':test_y,
            'pred':pred_y,
            'rty':rty
        })
    if config['return']:
        return metas
    else:
        tune.report(
            mae=np.mean(test_maes),
            std=np.std(test_maes),
            val_loss=np.mean(val_losses),
            rty=np.mean(rescore_maes),
        )

def plot_error(name,pred,test_label):
    sns.kdeplot(x=test_label,y=pred-test_label,fill=True,cbar=True)
    plt.xlabel('Score')
    plt.ylabel('Diff')
    plt.title(f'{name}: Prediction error distribution')
 
if __name__ =="__main__":
    matplotlib.use('AGG')
    parser = ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('--nfold',default=5,type=int)
    args = parser.parse_args()
    name=args.data.split('/')[-1]
    process_data(args.data,args.nfold,name)
    ray.init()

    analysis = tune.run(
        search_nnr,
        resources_per_trial={"cpu": 1, "gpu": 0.2},
        config={
        "lr": tune.grid_search([0.0001, 0.001, 0.01]),
        "dropout": tune.grid_search([0, 0.1, 0.3, 0.5]),
        "unit": tune.grid_search([16, 32, 64, 128]),
        'data':os.path.join(args.data+'_cv',f'cv'),
        'epoch':256,
        'return':False
    })
    best_config=analysis.get_best_trial(metric="val_loss", mode="min")
    config=best_config.config

    log_path=os.path.join(os.environ['PT_OUTPUT_DIR'],f'{name}.json')
    json.dump(best_config.last_result,open(log_path,'w'))
    
    config['return']=True
    for i,meta in enumerate(search_nnr(config)):
        plt.figure()
        plot_error(name,meta['pred'],meta['gt'])
        fig_path=os.path.join(os.environ['PT_OUTPUT_DIR'],f'{name}_{i+1}.png')
        plt.savefig(fig_path,dpi=200)

