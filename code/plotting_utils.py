import pandas as pd 
import numpy as np
from pathlib import Path
import math
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.optimize import minimize
from utils import *
from scipy.stats import multivariate_normal

def plotting_weights(w, sessInd, axes, trueW=None, title='', save_fig=False):
    ''' 
    Parameters
    __________
    w: N x K x D x C numpy array
        weight matrix (weights are fixed within one session)
    sessInd:
    title: str
        title of the plot
    Returns
    ________
    '''
    sess = len(sessInd)-1

    if (w.shape[1]==2):
        axes.plot(range(1,sess+1),w[sessInd[:-1],0,1,0],color='blue',marker='o',label='state 1 sensory')
        axes.plot(range(1,sess+1),w[sessInd[:-1],0,0,0],color='dodgerblue',marker='o', label='state 1 bias')
        axes.plot(range(1,sess+1),w[sessInd[:-1],1,1,0],color='darkred',marker='o',label='state 2 sensory')
        axes.plot(range(1,sess+1),w[sessInd[:-1],1,0,0],color='indianred',marker='o',label='state 2 bias')
        

        if(trueW is not None):
            axes.plot(range(1,sess+1),trueW[sessInd[:-1],0,1,0],color='blue',marker='o',linestyle='dashed', label='true sensory 1')
            axes.plot(range(1,sess+1),trueW[sessInd[:-1],0,0,0],color='dodgerblue',marker='o',linestyle='dashed', label='true bias 1')
            axes.plot(range(1,sess+1),trueW[sessInd[:-1],1,1,0],color='darkred',marker='o',linestyle='dashed', label='true sensory 2')
            axes.plot(range(1,sess+1),trueW[sessInd[:-1],1,0,0],color='indianred',marker='o',linestyle='dashed', label='true bias 2')
            
        axes.set_title(title)
        axes.set_xticks(range(1,sess+1))
        axes.set_ylabel("weights")
        axes.set_xlabel('session')
        axes.legend()
        #plt.legend(fontsize='xx-small')
    elif (w.shape[1]==1):
        axes.plot(range(1,sess+1),w[sessInd[:-1],0,1,0],color='blue',marker='o',label='state 1 sensory')
        axes.plot(range(1,sess+1),w[sessInd[:-1],0,0,0],color='dodgerblue',marker='o', label='state 1 bias')
        
        if(trueW is not None):
            axes.plot(range(1,sess+1),trueW[sessInd[:-1],0,1,0],color='blue',linestyle='dashed', label='true sensory')
            axes.plot(range(1,sess+1),trueW[sessInd[:-1],0,0,0],color='dodgerblue',linestyle='dashed', label='true bias')
            
        axes.set_title(title)
        axes.set_xticks(range(1,sess+1))
        axes.set_ylabel("weights")
        axes.set_xlabel('session')
        axes.legend(loc='upper left')
        #plt.legend(fontsize='xx-small')
    
    if(save_fig==True):
        plt.savefig(f'../figures/Weights_-{title}.png', bbox_inches='tight', dpi=400)

def sigma_testLl_plot(K, sigmaList, testLl, axes, title='', labels=None, save_fig=False):
    ''' 
    function for plotting the test LL vs sigma scalars
    '''
    inits = testLl.shape[0] # for mutiple initiaizations/models 
    colormap = sns.color_palette("viridis")
    sigmaListEven = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==0]
    sigmaListOdd = [sigmaList[ind] for ind in range(len(sigmaList)-4) if ind%2==1] + [sigmaList[ind] for ind in range(17,len(sigmaList))]
    flag = 0
    if (labels is None):
        labels = ['' for init in range(0,inits)]
        flag = 1
    for init in range(0,inits):
        axes.set_title(title)
        axes.scatter(np.log(sigmaList[1:]), testLl[init,1:], color=colormap[init])
        axes.plot(np.log(sigmaList[1:]), testLl[init,1:], color=colormap[init])
        if(sigmaList[0]==0):
            axes.scatter(-1 + np.log(sigmaList[1]), testLl[init,0], color=colormap[init], label=labels[init])
            if (K==1):
                axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM'] + [f'{np.round(sigma,3)}' for sigma in sigmaListOdd])
            else:
                axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM-HMM'] + [f'{np.round(sigma,3)}' for sigma in sigmaListOdd])
        else:
            axes.scatter(np.log(sigmaList[0]), testLl[init,0], color=colormap[init], label=f'init {init}')
            axes.set_xticks([np.log(sigmaListEven)],[f'{np.round(sigma,2)}' for sigma in sigmaListEven])
    axes.set_ylabel("Test LL (per trial)")
    axes.set_xlabel("sigma")
    if (flag == 0):
        axes.legend()

    if(save_fig==True):
        plt.savefig(f'../figures/Sigma_vs_TestLl-{title}.png', bbox_inches='tight', dpi=400)
    
def sigma_CV_testLl_plot_PWM(rat_id, stage_filter, K, folds, sigmaList, axes, title='', save_fig=False):
    ''' 
    function for plotting the test LL vs sigma scalars for PWM real data
    '''     
    
    colormap = sns.color_palette("viridis")
    for fold in range(0, folds):
        testLl = np.load(f'../data/testLl_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas.npy')
        axes.set_title(title)
        axes.scatter(np.log(sigmaList[1:]), testLl[1:], color=colormap[fold])
        axes.plot(np.log(sigmaList[1:]), testLl[1:], color=colormap[fold])
        if(sigmaList[0]==0):
            axes.scatter(-1 + np.log(sigmaList[1]), testLl[0], color=colormap[fold], label=f'fold {fold}')
            if (K==1):
                axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaList[1:])),['GLM'] + [f'{np.round(sigma,3)}' for sigma in sigmaList[1:]])
            else:
                axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaList[1:])),['GLM-HMM'] + [f'{np.round(sigma,3)}' for sigma in sigmaList[1:]])
        else:
            axes.scatter(np.log(sigmaList[0]), testLl[0], color=colormap[fold], label=f'fold {fold}')
            axes.set_xticks([np.log(sigmaList)],[f'{np.round(sigma,2)}' for sigma in sigmaList])
        axes.set_ylabel("Test LL (per trial)")
        axes.set_xlabel("sigma")
    axes.legend()

    if(save_fig==True):
        plt.savefig(f'../figures/Sigma_vs_TestLl-{title}.png', bbox_inches='tight', dpi=300)

    