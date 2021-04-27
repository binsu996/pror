'''
simluate under gauss error distribution
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from argparse import ArgumentParser
import seaborn as sns
import pandas as pd

def get_args():
    parser=ArgumentParser()
    parser.add_argument('error_type',choices=['normal','uniform'])
    parser.add_argument('output_name')
    return parser.parse_args()

def sim(upper_bound,error_type,error_param,size):
    objectives=np.random.uniform(0,upper_bound,size=size)
    if error_type=='normal':
        errors=np.random.normal(0,error_param,size=size)
    else:
        errors=np.random.uniform(-error_param,error_param,size=size)
    subjectives=(objectives+errors).clip(0,upper_bound)
    compare_matrix=(subjectives.reshape(-1,1)>subjectives).astype(float)
    fixed_subjectives=compare_matrix.sum(axis=-1)/size*upper_bound
    fixed_errors=fixed_subjectives-objectives
    return np.abs(errors).mean()-np.abs(fixed_errors).mean()

if __name__ == "__main__":
    args=get_args()
    matplotlib.use('AGG')
    grid=[]
    x_fn=lambda s:list(range(100,2001,s))
    y_fn=lambda s:list(range(0,51,s))
    for error_param in y_fn(1):
        row=[]
        for size in x_fn(1):
            r=sim(100,args.error_type,error_param,size)
            row.append(r)
        grid.append(row)
    X, Y = np.meshgrid(x_fn(1),y_fn(1))
    plt.contourf(X, Y, grid, cmap= 'Spectral')
    plt.xlabel('Number of Samples')
    plt.ylabel('M')
    plt.colorbar()
    plt.savefig(args.output_name,dpi=200)

    


