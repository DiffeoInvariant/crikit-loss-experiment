#!/usr/bin/python3
import pandas as pd
import sys
import matplotlib.pyplot as plt
import argparse
import numpy as np

def get_data(filename: str, std: str):
    data = pd.read_csv(filename)
    last_row = data.iloc[[-1]]
    niter = last_row.iloc[0]['step']
    final_loss = last_row.iloc[0]['loss']
    learned_a, learned_p = last_row.iloc[0]['param_0'], last_row.iloc[0]['param_1']
    final_grad = np.array([last_row.iloc[0]['grad_param_0'], last_row.iloc[0]['grad_param_1']])
    return float(std), niter, final_loss, np.array([learned_a,learned_p]), final_grad


def plot_niter_vs_std(l2_data, w1_data, method):
    stds = []
    l2_iters = []
    w1_iters = []
    for l, w in zip(l2_data, w1_data):
        if l[0] != w[0]:
            raise ValueError(f"Mismatch in stds in entries {l} and {w} of {l2_data} and {w1_data}")
        
        stds.append(l[0])
        l2_iters.append(l[1])
        w1_iters.append(w[1])
        
    plt.figure()
    plt.title(f'Number of {method} Iterations To Convergence vs Noise Level')
    plt.scatter(stds, l2_iters, s=10, marker='|', color='r', label='$L^2$')
    plt.scatter(stds, w1_iters, color='b', s=10, marker='_', label='$W^2$')
    plt.xlabel('Standard Deviation of Additive Gaussian Noise')
    plt.ylabel('Number of Iterations')
    plt.legend()


def plot_error_vs_std(l2_data, w1_data, method, true_val=np.array([1.0,1.2])):
    stds = []
    l2e = []
    w1e = []
    w1means = {}
    l2means = {}
    ns = {}
    for l, w in zip(l2_data, w1_data):
        if l[0] != w[0]:
            raise ValueError(f"Mismatch in stds in entries {l} and {w} of {l2_data} and {w1_data}")
        
        stds.append(l[0])
        std = l[0]
        wdiff = np.linalg.norm(true_val - w[3])
        ldiff = np.linalg.norm(true_val - l[3])
        if std in w1means:
            ns[std] += 1
            w1means[std] += wdiff
            l2means[std] += ldiff
        else:
            ns[std] = 1
            w1means[std] = wdiff
            l2means[std] = ldiff
        l2e.append(ldiff)
        w1e.append(wdiff)

    w1mu = []
    l2mu = []
    for std in stds:
        w1mu.append(w1means[std] / ns[std])
        l2mu.append(l2means[std] / ns[std])
    plt.figure()
    plt.title(r"$||\theta - \hat{\theta}||$ vs Noise Level")
    plt.scatter(stds, l2e,  color='firebrick', marker='|', label='$L^2$')
    plt.scatter(stds, w1e,  color='darkslategrey',  marker='_', label='$W^2$')
    plt.scatter(stds, l2mu, s=30, marker='|', color='magenta', label='$L^2$ Mean')
    plt.scatter(stds, w1mu, s=30, marker='_', color='lime', label='$W^2$ Mean')
    plt.xlabel('Standard Deviation of Additive Gaussian Noise')
    plt.ylabel(r'$||\theta - \hat{\theta}||$')
    plt.legend(bbox_to_anchor=(0.05,1),loc='upper left')

def get_filename(loss, observer, method, std, seed,  mode):
    return 'data/' + loss + '_' + observer + method + '_std' + std + '_seed_' + seed + '_' + mode + '_opt.csv'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Compare L^2 and W_2 to each other in a plot')
    parser.add_argument('-stds', '--stdevs', nargs='+', type=str, help='Standard deviations of the noise')
    parser.add_argument('-mode',choices=['form_invariant','energy'],default='form_invariant',help='What type of CR was used?')
    parser.add_argument('-oa','--opt_algo', type=str, default='L-BFGS-B', help='The name of the optimization algorithm')
    parser.add_argument('-obs', '--observer', type=str, default='outflow', help='What kind of observer?')
    parser.add_argument('-s', '--seeds', type=str, nargs='+', help='The random seeds')
    parser.add_argument('-plot_niter', action='store_true', default=False, help='Plot the number of iterations?')
    args = parser.parse_args()

    w2_data = []
    l2_data = []
    for std in args.stdevs:
        for seed in args.seeds:
            try:
                w2_data.append(get_data(get_filename('W2',args.observer,args.opt_algo,std,seed,args.mode), std))
            except:
                print("Error getting data for ", get_filename('W2',args.observer,args.opt_algo,std,seed,args.mode))
                raise
            try:
                l2_data.append(get_data(get_filename('L2',args.observer,args.opt_algo,std,seed,args.mode), std))
            except:
                print("Error getting data for ", get_filename('L2',args.observer,args.opt_algo,std,seed,args.mode))
                raise

    
    
    plot_error_vs_std(l2_data, w2_data, args.opt_algo)
    if args.plot_niter:
        plot_niter_vs_std(l2_data, w2_data, args.opt_algo)

    plt.show()
    second_seedlist = [str(2*int(x) - 153245) for x in args.seeds]
    w2_data = []
    l2_data = []
                       
    for seed in args.seeds + second_seedlist:
        for std in ['0.1', '0.25', '0.35', '0.45', '0.5', '0.55', '0.65', '0.7', '0.8', '1.0']:
            try:
                w2_data.append(get_data(get_filename('W2',args.observer,args.opt_algo,std,seed,args.mode), std))
            except:
                print("Error getting data for ", get_filename('W2',args.observer,args.opt_algo,std,seed,args.mode))
                raise
            try:
                l2_data.append(get_data(get_filename('L2',args.observer,args.opt_algo,std,seed,args.mode), std))
            except:
                print("Error getting data for ", get_filename('L2',args.observer,args.opt_algo,std,seed,args.mode))
                raise

    plot_error_vs_std(l2_data, w2_data, args.opt_algo)
    plt.show()
