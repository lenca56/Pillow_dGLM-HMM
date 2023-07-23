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

colors_dark = ['darkblue','darkred','darkgreen','darkgoldenrod']
colors_light = ['royalblue','indianred','limegreen','gold']

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
    for i in range(0,w.shape[1]):
        axes.plot(range(1,sess+1),w[sessInd[:-1],i,1,1],color=colors_dark[i],marker='o',label=f'state {i+1} sensory')
        axes.plot(range(1,sess+1),w[sessInd[:-1],i,0,1],color=colors_light[i],marker='o', label=f'state {i+1} bias')

    axes.set_title(title)
    axes.set_xticks(range(1,sess+1))
    axes.set_ylabel("weights")
    axes.set_xlabel('session')
    axes.legend()

    if(trueW is not None):
        for i in range(0,trueW.shape[1]):
            axes.plot(range(1,sess+1),trueW[sessInd[:-1],i,1,1],color=colors_dark[i],marker='o',linestyle='dashed', label=f'true sensory {i+1}')
            axes.plot(range(1,sess+1),trueW[sessInd[:-1],i,0,1],color=colors_light[i],marker='o',linestyle='dashed', label=f'true bias {i+1}')
    
    if(save_fig==True):
        plt.savefig(f'../figures/Weights_-{title}.png', bbox_inches='tight', dpi=400)

def plotting_transition_matrix_stickiness(p, sessInd, axes, trueP=None, title='', save_fig=False):
    ''' 
    
    Parameters
    __________
    
    Returns
    ________
    '''

    sess = len(sessInd)-1
    for i in range(0,p.shape[1]):
        axes.plot(range(1,sess+1),p[sessInd[:-1],i,i],color=colors_dark[i],marker='o',label=f'state {i+1} stickiness')

    axes.set_title(title)
    axes.set_xticks(range(1,sess+1))
    axes.set_ylabel("P( state t+1 = i | state t = i )")
    axes.set_ylim(0.6,1)
    axes.set_xlabel('session')
    axes.legend(loc='lower right')

    if(trueP is not None):
        for i in range(0,p.shape[1]):
            axes.plot(range(1,sess+1),trueP[sessInd[:-1],i,i],color=colors_dark[i],marker='o',linestyle='dashed',label=f'true state {i+1} ')

    if(save_fig==True):
        plt.savefig(f'../figures/TransitionMatrix_stickiness_-{title}.png', bbox_inches='tight', dpi=400)

def plotting_state_occupancy(z, axes, title='', save_fig=False):
    K = len(np.unique(z)) # number of states
    percent_time = np.zeros((K,))
    labels = ['state %s' %(i+1) for i in range(K)]
    for i in range(0,K):
        percent_time[i] = len(np.argwhere(z==i))/z.shape[0]

    axes.bar(labels, percent_time, color=colors_dark)
    axes.set_ylim(0,1)
    axes.set_title(title)
    axes.set_ylabel('% time in state')
    if(save_fig==True):
        plt.savefig(f'../figures/State_Occupancy-{title}.png', bbox_inches='tight', dpi=400)

def sigma_testLl_plot(K, sigmaList, testLl, axes, title='', labels=None, color=0, save_fig=False):
    ''' 
    function for plotting the test LL vs sigma scalars
    '''
    inits = testLl.shape[0] # for mutiple initiaizations/models 
    colormap = sns.color_palette("viridis")
    sigmaListEven = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==0]
    if (len(sigmaList)>=17):
        sigmaListOdd = [sigmaList[ind] for ind in range(len(sigmaList)-4) if ind%2==1] + [sigmaList[ind] for ind in range(17,len(sigmaList))]
    else:
        sigmaListOdd = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==1]
    flag = 0
    if (labels is None):
        labels = ['' for init in range(0,inits)]
        flag = 1
    for init in range(0,inits):
        axes.set_title(title)
        axes.scatter(np.log(sigmaList[1:]), testLl[init,1:], color=colormap[color])
        axes.plot(np.log(sigmaList[1:]), testLl[init,1:], color=colormap[color])
        if(sigmaList[0]==0):
            axes.scatter(-1 + np.log(sigmaList[1]), testLl[init,0], color=colormap[color], label=labels[init])
            if (K==1):
                axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM'] + [f'{np.round(sigma,3)}' for sigma in sigmaListOdd])
            else:
                axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM-HMM'] + [f'{np.round(sigma,3)}' for sigma in sigmaListOdd])
        else:
            axes.scatter(np.log(sigmaList[0]), testLl[init,0], color=colormap[color], label=f'init {init}')
            axes.set_xticks([np.log(sigmaListEven)],[f'{np.round(sigma,2)}' for sigma in sigmaListEven])
    axes.set_ylabel("Test LL (per trial)")
    axes.set_xlabel("sigma")
    if (flag == 0):
        axes.legend()

    if(save_fig==True):
        plt.savefig(f'../figures/Sigma_vs_TestLl-{title}.png', bbox_inches='tight', dpi=400)
    
def sigma_CV_testLl_plot_PWM(rat_id, stage_filter, K, folds, sigmaList, axes, title='', labels=None, color=0, penaltyW=False, save_fig=False):
    ''' 
    function for plotting the test LL vs sigma scalars for PWM real data
    '''     
    
    colormap = sns.color_palette("viridis")
    sigmaListEven = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==0]
    sigmaListOdd = [sigmaList[ind] for ind in range(11) if ind%2==1] + [sigmaList[ind] for ind in range(11,len(sigmaList))]
    for fold in range(0, folds):
        testLl = np.load(f'../data_PWM/testLl_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_penaltyW={penaltyW}.npy')
        axes.set_title(title)
        axes.scatter(np.log(sigmaList[1:]), testLl[1:], color=colormap[color+fold], label=labels[fold])
        axes.plot(np.log(sigmaList[1:]), testLl[1:], color=colormap[color+fold])
        if(sigmaList[0]==0):
            axes.scatter(-2 + np.log(sigmaList[1]), testLl[0], color=colormap[color+fold])
            if (K==1):
                axes.set_xticks([-2 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM'] + [f'{np.round(sigma,4)}' for sigma in sigmaListOdd])
            else:
                axes.set_xticks([-2 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM-HMM'] + [f'{np.round(sigma,4)}' for sigma in sigmaListOdd])
        else:
            axes.scatter(np.log(sigmaList[0]), testLl[0], color=colormap[color+fold])
            axes.set_xticks([np.log(sigmaListEven)],[f'{np.round(sigma,4)}' for sigma in sigmaListEven])
        axes.set_ylabel("Test LL (per trial)")
        axes.set_xlabel("sigma")
    axes.legend()

    if(save_fig==True):
        plt.savefig(f'../figures/Sigma_vs_TestLl-{title}.png', bbox_inches='tight', dpi=400)

def plotting_weights_PWM(w, sessInd, axes, sessStop=None, title='', save_fig=False):
    K = w.shape[1]
    sess = len(sessInd)-1

    if (K==1):
        if (sessStop==None):
            axes[1].plot(range(1,sess+1),w[sessInd[:-1],0,4,1],color='#59C3C3', linewidth=5, label='previous choice', alpha=0.8)
            axes[1].plot(range(1,sess+1),w[sessInd[:-1],0,5,1],color='#9593D9',linewidth=5, label='previous correct', alpha=0.8)
            axes[1].plot(range(1,sess+1),w[sessInd[:-1],0,3,1],color='#99CC66',linewidth=5, label='previous stim', alpha=0.8)
            axes[0].plot(range(1,sess+1),w[sessInd[:-1],0,1,1],color="#A9373B",linewidth=5,label='stim A', alpha=0.8)
            axes[0].plot(range(1,sess+1),w[sessInd[:-1],0,0,1],color='#FAA61A',linewidth=5, label='bias', alpha=0.8)
            axes[0].plot(range(1,sess+1),w[sessInd[:-1],0,2,1],color="#2369BD",linewidth=5, label='stim B', alpha=0.8)
        else:
            axes[1].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,4,1],color='#59C3C3', linewidth=5, label='previous choice', alpha=0.8)
            axes[1].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,5,1],color='#9593D9',linewidth=5, label='previous correct', alpha=0.8)
            axes[1].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,3,1],color='#99CC66',linewidth=5, label='previous stim', alpha=0.8)
            axes[0].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,1,1],color="#A9373B",linewidth=5,label='stim A', alpha=0.8)
            axes[0].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,0,1],color='#FAA61A',linewidth=5, label='bias', alpha=0.8)
            axes[0].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,2,1],color="#2369BD",linewidth=5, label='stim B', alpha=0.8)

        #axes[0].set_title(title)
        axes[0].set_ylabel("weights")
        axes[0].set_xlabel('session')
        # axes[0].set_yticks([-2,0,2])
        # axes[0].set_ylim(-3,3)
        # axes[1].set_ylim(-0.2,2.1)
        # axes[1].set_yticks([0,2])
        #axes[1].set_ylabel("weights")
        axes[1].set_xlabel('session')

        axes[0].legend(loc='upper right')
        axes[1].legend(loc='upper right')

    elif(K >= 2):
        for i in range(0,K):
            if (sessStop==None):
                axes[i,1].plot(range(1,sess+1),w[sessInd[:-1],i,4,1],color='#59C3C3', linewidth=5, label='previous choice', alpha=0.8)
                axes[i,1].plot(range(1,sess+1),w[sessInd[:-1],i,5,1],color='#9593D9',linewidth=5, label='previous correct', alpha=0.8)
                axes[i,1].plot(range(1,sess+1),w[sessInd[:-1],i,3,1],color='#99CC66',linewidth=5, label='previous stim', alpha=0.8)
                axes[i,0].plot(range(1,sess+1),w[sessInd[:-1],i,1,1],color="#A9373B",linewidth=5,label='stim A', alpha=0.8)
                axes[i,0].plot(range(1,sess+1),w[sessInd[:-1],i,0,1],color='#FAA61A',linewidth=5, label='bias', alpha=0.8)
                axes[i,0].plot(range(1,sess+1),w[sessInd[:-1],i,2,1],color="#2369BD",linewidth=5, label='stim B', alpha=0.8)
            else:
                axes[i,1].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,4,1],color='#59C3C3', linewidth=5, label='previous choice', alpha=0.8)
                axes[i,1].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,5,1],color='#9593D9',linewidth=5, label='previous correct', alpha=0.8)
                axes[i,1].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,3,1],color='#99CC66',linewidth=5, label='previous stim', alpha=0.8)
                axes[i,0].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,1,1],color="#A9373B",linewidth=5,label='stim A', alpha=0.8)
                axes[i,0].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,0,1],color='#FAA61A',linewidth=5, label='bias', alpha=0.8)
                axes[i,0].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,2,1],color="#2369BD",linewidth=5, label='stim B', alpha=0.8)

            axes[i,0].set_title(f'State {i+1}')
            axes[i,1].set_title(f'State {i+1}')
            axes[i,0].set_ylabel("weights")
            # axes[i,0].set_yticks([-2,0,2])
            # axes[i,0].set_ylim(-3,3)
            # axes[i,1].set_ylim(-0.2,2.1)
            # axes[i,1].set_yticks([0,2])
            #axes[1].set_ylabel("weights")
            if (i==K-1):
                axes[i,0].set_xlabel('session')
                axes[i,1].set_xlabel('session')
            axes[i,0].legend(loc='upper right')
            axes[i,1].legend(loc='upper right')

    if(save_fig==True):
        plt.savefig(f'../figures/Weights_PWM_{title}.png', bbox_inches='tight', dpi=400)


    