import pandas as pd
from argparse import ArgumentParser
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np

def get_args():
    parser=ArgumentParser()
    parser.add_argument('distribution_src')
    parser.add_argument('--plot',action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args=get_args()
    data=pd.read_csv(args.distribution_src,sep='\t',names=['r1','r2','r3','r4'],index_col=0)

    # ouput raters' distribution
    if args.plot:
        plt.clf()
        sns.kdeplot(data=data,common_norm=False)
        plt.xlabel('Score')
        plt.title('Subjective score distributions')
        plt.savefig('/mnt/tmp/rater_distributions.png',dpi=200)

    mean=data.mean(axis=1)
    mae=data.mad(axis=1)
    data['mean']=mean
    for i in range(1,5):
        data[f'e{i}']=data[f'r{i}']-data['mean']
        p=np.corrcoef(data[f'r{i}'],data['mean'])[0][1]
        print(f'{i}-th rater: pearson {p:.3f}')

        if args.plot:
            plt.clf()
            sns.kdeplot(x=data['mean'],y=data[f'e{i}'],fill=True,cbar=True)
            plt.xlabel('objective score')
            plt.ylabel('error')
            plt.title(f'r{i}_error_distribution')
            plt.savefig(f'/mnt/tmp/r{i}_error_distribution.png',dpi=200)

    data['mae']=mae

    if args.plot:
        plt.clf()
        sns.kdeplot(data=data[[f'e{i}' for i in range(1,5)]],common_norm=False)
        plt.xlabel('Score Error')
        plt.title('Subjective error distributions')
        plt.savefig('/mnt/tmp/rater_error_distributions.png',dpi=200)
    

    mean_gmm=GaussianMixture(n_components=6).fit(np.reshape(data['mean'].values,(-1,1)))
    data['resample_mean']=np.array(mean_gmm.sample(2511)[0]).reshape(-1).clip(0,5)

    if args.plot:
        plt.clf()
        sns.kdeplot(data=data[['mean','resample_mean']],common_norm=False)
        plt.xlabel('Score')
        plt.title('Resample data distribution')
        plt.savefig('/mnt/tmp/resample_data_distributions.png',dpi=200)

    for i in range(1,5):
        gmm=GaussianMixture(n_components=1).fit(np.reshape(data[f'e{i}'].values,(-1,1)))
        data[f'resample_e{i}']=np.array(gmm.sample(2511)[0]).reshape(-1)
        if args.plot:
            plt.clf()
            sns.kdeplot(data=data[[f'e{i}',f'resample_e{i}']],common_norm=False)
            plt.xlabel('Score')
            plt.title('Resample error distribution')
            plt.savefig(f'/mnt/tmp/resample_error{i}_distributions.png',dpi=200)

    objective=data['mean'].values
    sorted_objective=np.sort(objective)
    for i in range(1,5):
        subjective=data[f'r{i}'].values
        fixed_subjective=np.sort(objective)[np.argsort(np.argsort(subjective))]
        subjective_mae=np.mean(np.abs(subjective-objective))
        fixed_subjective_mae=np.mean(np.abs(fixed_subjective-objective))
        print(f'{i}-th rater: mae {subjective_mae:.3f} -> {fixed_subjective_mae:.3f}')

    objective=data['resample_mean'].values
    sorted_objective=np.sort(objective)
    for i in range(1,5):
        subjective_error=data[f'resample_e{i}'].values
        fixed_subjective=np.sort(objective)[np.argsort(np.argsort(subjective_error+objective))]
        subjective_mae=np.mean(np.abs(subjective_error))
        fixed_subjective_mae=np.mean(np.abs(fixed_subjective-objective))
        print(f'resample {i}-th rater: mae {subjective_mae:.3f} -> {fixed_subjective_mae:.3f}')

    # print(data)
