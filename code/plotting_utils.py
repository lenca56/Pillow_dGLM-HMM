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

def plotting_weights(w, sessInd, trueW=None, title=''):
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
        plt.plot(range(1,sess+1),w[sessInd[:-1],0,1,0],color='blue',marker='o',label='state 0 sensory')
        plt.plot(range(1,sess+1),w[sessInd[:-1],0,0,0],color='dodgerblue',marker='o', label='state 0 bias')
        plt.plot(range(1,sess+1),w[sessInd[:-1],1,1,0],color='darkred',marker='o',label='state 1 sensory')
        plt.plot(range(1,sess+1),w[sessInd[:-1],1,0,0],color='indianred',marker='o',label='state 1 bias')
        

        if(trueW is not None):
            plt.plot(range(1,sess+1),trueW[sessInd[:-1],0,1,0],color='blue',marker='o',linestyle='dashed', label='true sensory 0')
            plt.plot(range(1,sess+1),trueW[sessInd[:-1],0,0,0],color='dodgerblue',marker='o',linestyle='dashed', label='true bias 0')
            plt.plot(range(1,sess+1),trueW[sessInd[:-1],1,1,0],color='darkred',marker='o',linestyle='dashed', label='true sensory 1')
            plt.plot(range(1,sess+1),trueW[sessInd[:-1],1,0,0],color='indianred',marker='o',linestyle='dashed', label='true bias 1')
            
        plt.title(title)
        plt.xticks(range(1,sess+1))
        plt.ylabel("weights")
        plt.title(title)
        plt.xlabel('session')
        plt.legend()
        #plt.legend(fontsize='xx-small')
        plt.show()
    elif (w.shape[1]==1):
        plt.plot(range(1,sess+1),w[sessInd[:-1],0,1,0],color='blue',marker='o',label='state 0 sensory')
        plt.plot(range(1,sess+1),w[sessInd[:-1],0,0,0],color='dodgerblue',marker='o', label='state 0 bias')
        
        if(trueW is not None):
            plt.plot(range(1,sess+1),trueW[sessInd[:-1],0,1,0],color='blue',linestyle='dashed', label='true sensory')
            plt.plot(range(1,sess+1),trueW[sessInd[:-1],0,0,0],color='dodgerblue',linestyle='dashed', label='true bias')
            
        plt.title(title)
        plt.xticks(range(1,sess+1))
        plt.ylabel("weights")
        plt.title(title)
        plt.xlabel('session')
        plt.legend()
        #plt.legend(fontsize='xx-small')
        plt.show()

def sigma_testLl_plot(sigmaList, testLl, axes, title='', label='', save_fig=False):
    ''' 
    function for plotting the test LL vs sigma scalars
    '''
    inits = testLl.shape[0]
    colormap = sns.color_palette("viridis")
    for init in range(0,inits):
        axes.set_title(title)
        axes.scatter(np.log(sigmaList[1:]), testLl[init,1:], color=colormap[init])
        axes.plot(np.log(sigmaList[1:]), testLl[init,1:], color=colormap[init])
        if(sigmaList[0]==0):
            axes.scatter(-1 + np.log(sigmaList[1]), testLl[init,0], color=colormap[init], label=f'init {init}')
            if (K==1):
                axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaList[1:])),['GLM'] + [f'{np.round(sigma,3)}' for sigma in sigmaList[1:]])
            else:
                axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaList[1:])),['GLM-HMM'] + [f'{np.round(sigma,3)}' for sigma in sigmaList[1:]])
        else:
            axes.scatter(np.log(sigmaList[0]), testLl[init,0], color=colormap[init], label=f'init {init}')
            axes.set_xticks([np.log(sigmaList)],[f'{np.round(sigma,2)}' for sigma in sigmaList])
        axes.set_ylabel("Test LL (per trial)")
        axes.set_xlabel("sigma")
        #axes.legend()

    if(save_fig==True):
        plt.savefig(f'../figures/Sigma_vs_TestLl-{title}', bbox_inches='tight', dpi=300)
    
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

    