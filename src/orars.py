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

def process_data(filename,nfold):
    dataset=pd.read_csv(filename,sep=',\s*|\s+',header=None,index_col=None)
    tmp=dataset.values[:,-1]
    # print(len(tmp))
    # print(min(tmp),max(tmp))
    # exit()
    cv=CrossValidation(dataset.values,nfold,shuffle_seed=666,split_seed=888)
    data=[x for x in cv]
    dir_path=filename+'_cv'
    os.makedirs(dir_path,exist_ok=True)
    cv_data=[]
    for i in range(len(data)):
        dataset.to_csv(os.path.join(dir_path,'data'))
        train,valid,test=data[i]
        pd.DataFrame(data=train).to_csv(os.path.join(dir_path,f'train.{i}'),header=False,index=False)
        pd.DataFrame(data=valid).to_csv(os.path.join(dir_path,f'valid.{i}'),header=False,index=False)
        pd.DataFrame(data=test).to_csv(os.path.join(dir_path,f'test.{i}'),header=False,index=False)
        train_valid_test=[train[:,:-1],train[:,-1],valid[:,:-1],valid[:,-1],test[:,:-1],test[:,-1]]
        pickle.dump(train_valid_test,open(os.path.join(dir_path,f'cv.{i}'),'wb'))
        cv_data.append(train_valid_test)
    pickle.dump(cv_data,open(os.path.join(dir_path,f'cv'),'wb'))
    print('Data processing is OK!')

def load_data(path):
    return pickle.load(open(path,'rb'))

class OrdRegNet(nn.Module):
    def __init__(self, units, dropout):
        super(OrdRegNet, self).__init__()
        layers = [nn.BatchNorm1d(units[0])]
        layers=[]
        units = zip(units[:-1], units[1:])
        for in_dim, out_dim in units:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Dropout(p=dropout,inplace=True))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers[:-3]).cuda()

    def forward(self, x):
        return self.net(x)

def search_ORARS(config):
    cv_data=load_data(config['data'])
    unit=config['unit']
    dropout=config['dropout']
    evalutor=EvaluatorEX()
    val_losses=[]
    test_maes=[]

    for train_X,train_y,valid_X,valid_y,test_X,test_y in cv_data:
        if config['full']==False:
            train_X=train_X[:256]
            train_y=train_y[:256]
            valid_X=valid_X[:256]
            valid_y=valid_y[:256]
            batch_size=512
        else:
            batch_size=128

        org_config={
            'model_protocol':OrdRegNet,
            'model_args':{
                'units':[train_X.shape[1]*2,unit,unit,unit,2],
                'dropout':config['dropout']
            },
            'config':{
                'lr':config['lr'],
                'cuda':True,
                'generate_weight':True,
                'bin_width':10000,
                'batch_size':batch_size,
                'soft_scale':1,
                'epoch':config['epoch'],
            },
            'log_dir':f'exp/{uuid4()}',
            'ignore_check_dirs':False,
            'remove_old':True
        }
        org=OrdinalRegression(**org_config)
        anchor_set=list(zip(train_X,train_y))
        valid_set=list(zip(valid_X,valid_y))
        org.train(anchor_set,valid_set,show_time=True)
        val_losses.append(org.val_loss)
        print(org.val_loss)
        test_set=list(zip(test_X,test_y))
        preds=[]
        for i in range(0,len(test_set),64):
            sub_set=test_set[i:i+64]
            pred=org.eval(anchor_set,sub_set)[1]
            preds.append(pred)
        pred=np.concatenate(preds)
        
        mae=np.mean(np.abs(pred-test_y))
        test_maes.append(mae)

    if config['full']:
        print({
            'test_mae':np.mean(test_maes),
            'val_loss':np.mean(val_losses),
        })
    else:
        tune.report(
            mae=np.mean(test_maes),
            std=np.std(test_maes),
            val_loss=np.mean(val_losses),
        )
 
if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('--nfold',default=5,type=int)
    parser.add_argument('--aml',action='store_true')
    args = parser.parse_args()
    name=args.data.split('/')[-1]
    process_data(args.data,args.nfold)
    

    # analysis = tune.run(
    #     search_ORARS,
    #     resources_per_trial={"cpu": 1, "gpu": 1},
    #     reuse_actors=True,
    #     config={
    #         "lr": tune.grid_search([0.0001, 0.001, 0.01]),
    #         "dropout": tune.grid_search([0, 0.1, 0.3, 0.5]),
    #         "unit": tune.grid_search([16, 32, 64, 128]),
    #         'data':os.path.join(args.data+'_cv',f'cv'),
    #         'epoch':8,
    #         'full':False}
    # )
    # best_config=analysis.get_best_trial(metric="val_loss", mode="min")
    # config=best_config.config

    # log_path=os.path.join(os.environ['PT_OUTPUT_DIR'],f'{name}_orars.json')
    # json.dump(best_config.last_result,open(log_path,'w'))

    config={
        "lr": 0.001,
        "dropout": 0,
        "unit": 128,
        'data':os.path.join(args.data+'_cv',f'cv'),
        'epoch':16,
        'full':True
    }
    search_ORARS(config)
