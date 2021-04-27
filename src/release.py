import numpy as np
from torch import nn
from ordinalregression import CVTrial,Releaser,OrdinalRegression
import pickle
import sys
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import os
import torch
import pandas as pd
import json

class OrdRegNet(nn.Module):
    def __init__(self, units):
        super(OrdRegNet, self).__init__()
        layers = [nn.BatchNorm1d(units[0])]
        layers=[]
        units = zip(units[:-1], units[1:])
        for in_dim, out_dim in units:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.Dropout(p=0.1,inplace=True))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers[:-3]).cuda()

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('log')
    parser.add_argument('release_file')
    args = parser.parse_args()

    data=json.load(open(args.data,'r'))
    train=data[:-10]
    test=data[-10:]

    org_config={
        'model_protocol':OrdRegNet,
        'model_args':{
            'units':[262,128,128,128,2]
        },
        'config':{
            'lr':1e-4,
            'cuda':True,
            'generate_weight':True,
            'bin_width':3,
            'batch_size':1024,
            'soft_scale':0.5,
            'epoch':3,
        },
        'log_dir':args.log,
        'ignore_check_dirs':False,
        'remove_old':True
    }

    exp=Releaser(org_config)
    exp.run(train,test,args.release_file,'onnx')