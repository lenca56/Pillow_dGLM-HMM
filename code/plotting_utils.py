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


def plot_testLl_CV_sigma(testLl, sigmaList, label, color, axes, linestyle='-o'):
    '''  
    function that plots test LL as a function of sigma 

    Parameters
    ----------
    testLl: len(sigmaList) x 1 numpy array
        per trial log-likelihood on test data
    sigmaList: list

    '''
    colormap = sns.color_palette("viridis")
    sigmaListEven = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==0]
    sigmaListOdd = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==1] #+ [sigmaList[ind] for ind in range(11,len(sigmaList))]
    # sigmaListOdd = [sigmaList[ind] for ind in range(11) if ind%2==1] #+ [sigmaList[ind] for ind in range(11,len(sigmaList))]
    axes.plot(np.log(sigmaList[1:]), testLl[1:], linestyle, color=colormap[color], label=label)
    if(sigmaList[0]==0):
        axes.scatter(-2 + np.log(sigmaList[1]), testLl[0], color=colormap[color])
        axes.set_xticks([-2 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM-HMM'] + [f'{np.round(sigma,4)}' for sigma in sigmaListOdd])
    else:
        axes.scatter(np.log(sigmaList[0]), testLl[0], color=colormap[color])
        axes.set_xticks([np.log(sigmaListEven)],[f'{np.round(sigma,4)}' for sigma in sigmaListEven])
    axes.set_ylabel("Test LL (per trial)")
    axes.set_xlabel("sigma")
    axes.legend()

def plotting_weights_PWM(w, sessInd, axes, sessStop=None, yLim=[-3,3,-1,1], title='', save_fig=False):
    # permute weights accordinng to highest sensory
    sortedStateInd = get_states_order(w, sessInd)
    w = w[:,sortedStateInd,:,:]

    K = w.shape[1]
    sess = len(sessInd)-1

    if (K==1):
        axes[0].axhline(0, alpha=0.3, color='black',linestyle='--')
        axes[1].axhline(0, alpha=0.3, color='black',linestyle='--')
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
        axes[0].set_ylim(yLim[0:2])
        axes[1].set_ylim(yLim[2:4])
        # axes[1].set_yticks([0,2])
        #axes[1].set_ylabel("weights")
        axes[1].set_xlabel('session')

        axes[0].legend()
        axes[1].legend()

    elif(K >= 2):
        for i in range(0,K):
            axes[i,0].axhline(0, alpha=0.3, color='black',linestyle='--')
            axes[i,1].axhline(0, alpha=0.3, color='black',linestyle='--')
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
            axes[i,0].set_ylim(yLim[0:2])
            axes[i,1].set_ylim(yLim[2:4])
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

def plotting_weights_IBL(w, sessInd, axes, yLim, colors=None, labels=None):
    # permute weights accordinng to highest sensory
    sortedStateInd = get_states_order(w, sessInd)
    w = w[:,sortedStateInd,:,:]

    D = w.shape[2]

    K = w.shape[1]
    sess = len(sessInd)-1

    if (K==1):
        axes.axhline(0, alpha=0.3, color='black',linestyle='--')
        for d in range(0, D):
            axes.plot(range(1,sess+1),w[sessInd[:-1],0,d,1],color=colors[d],linewidth=5,label=labels[d], alpha=0.8)
        axes.set_ylabel("weights")
        axes.set_xlabel('session')
        axes.set_ylim(yLim)
        axes.set_title(f'State 1')
        axes.legend()
    else:
        for k in range(0,K):
            axes[k].axhline(0, alpha=0.3, color='black',linestyle='--')
            for d in range(0, D):
                axes[k].plot(range(1,sess+1),w[sessInd[:-1],k,d,1],color=colors[d],linewidth=5,label=labels[d], alpha=0.8)
            axes[k].set_ylim(yLim)
            axes[k].set_ylabel("weights")
            axes[k].set_title(f'State {k+1}')
            axes[k].legend()
        axes[K-1].set_xlabel('session')

def plotting_weights_per_feature(w, sessInd, axes, yLim=[-3,3], colors=colorsStates, labels=myFeatures):
    # permute weights accordinng to highest sensory
    sortedStateInd = get_states_order(w, sessInd)
    w = w[:,sortedStateInd,:,:]
    D = w.shape[2]
    K = w.shape[1]
    sess = len(sessInd)-1

    for d in range(0,D):
        axes[d].axhline(0, alpha=0.3, color='black',linestyle='--')
        for k in range(0, K):
            axes[d].plot(range(1,sess+1),w[sessInd[:-1],k,d,1],color=colors[k],linewidth=5,label=f'state {k+1}', alpha=0.8)
        axes[d].set_yticks(np.arange(-5,5,1))
        axes[d].set_ylim(yLim[d])
        axes[d].set_ylabel("weights")
        axes[d].set_title(f'{labels[d]}')
        axes[d].legend(loc = 'upper right')
    axes[D-1].set_xlabel('session')

def plot_transition_matrix(P, sortedStateInd):
    ''' 
    function that plots heatmap of transition matrix (assumed constant)

    Parameters
    ----------
    P: K x K numpy array
        transition matrix to be plotted 

    Returns
    ----------
    '''
    P = P[sortedStateInd,:][:,sortedStateInd]

    K = P.shape[0]
    s = sns.heatmap(np.round(P,3),annot=True,cmap='BuPu', fmt='g')
    s.set(xlabel='state at time t+1', ylabel='state at time t', title='Recovered transition matrix P', xticklabels=range(1,K+1), yticklabels=range(1,K+1))
    fig = s.get_figure()

def plot_posteior_latent(gamma, sessInd, sessions = [20,68,160]):
    stateColors = ['darkblue','green','orange']
    s = len(sessions)
    if (s>5):
        raise Exception("Cant have more than 5 example sessions to plot")
    K = gamma.shape[1]
    fig, axes = plt.subplots(3,1, figsize=(20,15))
    for i in range(0,s):
        axes[i].set_title('session ' + str(sessions[i]))
        axes[-1].set_xlabel('trials')
        axes[i].set_ylabel('posterior latent')
        for k in range(0,K):
            axes[i].plot(np.arange(sessInd[sessions[i]+1]-sessInd[sessions[i]]), gamma[sessInd[sessions[i]]:sessInd[sessions[i]+1],k], color=stateColors[k], label=f'state {k+1}')
        axes[i].legend()

from datetime import date, datetime, timedelta

def IBL_plot_performance(dfAll, subject, axes, sessStop=-1):
    # code from Psytrack
    p = 5
    df = dfAll[dfAll['subject']==subject]   # Restrict data to the subject specified
    cL = np.tanh(p*df['contrastLeft'])/np.tanh(p)   # tanh transformation of left contrasts
    cR = np.tanh(p*df['contrastRight'])/np.tanh(p)  # tanh transformation of right contrasts
    inputs = dict(cL = np.array(cL)[:, None], cR = np.array(cR)[:, None])

    outData = dict(
        subject=subject,
        lab=np.unique(df["lab"])[0],
        contrastLeft=np.array(df['contrastLeft']),
        contrastRight=np.array(df['contrastRight']),
        date=np.array(df['date']),
        dayLength=np.array(df.groupby(['date','session']).size()),
        correct=np.array(df['feedbackType']),
        correctSide=np.array(df['correctSide']),
        probL=np.array(df['probabilityLeft']),
        inputs = inputs,
        y = np.array(df['choice'])
    )

    easy_trials = (outData['contrastLeft'] > 0.45).astype(int) | (outData['contrastRight'] > 0.45).astype(int)
    perf = []
    length = []
    for d in np.unique(outData['date']):
        date_trials = (outData['date'] == d).astype(int)
        inds = (date_trials * easy_trials).astype(bool)
        perf += [np.average(outData['correct'][inds])]
        length.append((outData['date'] == d).sum())

    dates = np.unique([datetime.strptime(i, "%Y-%m-%d") for i in outData['date']])
    dates = np.arange(len(dates)) + 1

    # My plotting function

    l1, = axes[0].plot(dates[:sessStop], perf[:sessStop], color="black", linewidth=1.5, zorder=2) # only look at first 25 days
    l2, = axes[1].plot(dates[:sessStop], length[:sessStop], color='gray', linestyle='--')
    # plt.scatter(dates[9], perf[9], c="white", s=30, edgecolors="black", linestyle="--", lw=0.75, zorder=5, alpha=1) # first session >50% accuracy has circle

    axes[0].axhline(0.5, color="black", linestyle="-", lw=1, alpha=0.3, zorder=0)

    # axes[0].set_xticks(np.arange(0,sessStop+1,5))
    axes[0].set_yticks([0.4,0.6,0.8,1.0])
    axes[0].set_ylim(0.2,1.0)
    axes[1].set_ylim(100,1500)
    # axes[0].set_xlim(1, sessStop + .5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title('Accuracy and session length for IBL mouse ' + subject)
    axes[0].set_xlabel("days of training")
    axes[0].set_ylabel('accuracy on easy trials')
    axes[1].set_ylabel('number of trials')
    axes[0].legend([l1, l2], ["% correct", "# trials"])
    plt.subplots_adjust(0,0,1,1) 

# OLD FUNCTION
# def sigma_CV_testLl_plot_PWM(rat_id, stage_filter, K, folds, sigmaList, axes, title='', labels=None, color=0, linestyle='solid', penaltyW=False, save_fig=False):
#     ''' 
#     function for plotting the test LL vs sigma scalars for PWM real data
#     '''     
    
#     colormap = sns.color_palette("viridis")
#     sigmaListEven = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==0]
#     sigmaListOdd = [sigmaList[ind] for ind in range(11) if ind%2==1] + [sigmaList[ind] for ind in range(11,len(sigmaList))]
#     for fold in range(0, folds):
#         testLl = np.load(f'../data_PWM/testLl_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_penaltyW={penaltyW}.npy')
#         axes.set_title(title)
#         # axes.scatter(np.log(sigmaList[1:]), testLl[1:], color=colormap[color+fold])
#         axes.plot(np.log(sigmaList[1:]), testLl[1:], '-o', color=colormap[color+fold], linestyle=linestyle, label=labels[fold])
#         if(sigmaList[0]==0):
#             axes.scatter(-2 + np.log(sigmaList[1]), testLl[0], color=colormap[color+fold])
#             if (K==1):
#                 axes.set_xticks([-2 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM'] + [f'{np.round(sigma,4)}' for sigma in sigmaListOdd])
#             else:
#                 axes.set_xticks([-2 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM-HMM'] + [f'{np.round(sigma,4)}' for sigma in sigmaListOdd])
#         else:
#             axes.scatter(np.log(sigmaList[0]), testLl[0], color=colormap[color+fold])
#             axes.set_xticks([np.log(sigmaListEven)],[f'{np.round(sigma,4)}' for sigma in sigmaListEven])
#         axes.set_ylabel("Test LL (per trial)")
#         axes.set_xlabel("sigma")
#     axes.legend()

#     if(save_fig==True):
#         plt.savefig(f'../figures/Sigma_vs_TestLl-{title}.png', bbox_inches='tight', dpi=400)


# OLD FUNCTION
# def sigma_testLl_plot(K, sigmaList, testLl, axes, title='', labels=None, color=0, save_fig=False):
#     ''' 
#     function for plotting the test LL vs sigma scalars
#     '''
#     inits = testLl.shape[0] # for mutiple initiaizations/models 
#     colormap = sns.color_palette("viridis")
#     sigmaListEven = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==0]
#     if (len(sigmaList)>=17):
#         sigmaListOdd = [sigmaList[ind] for ind in range(len(sigmaList)-4) if ind%2==1] + [sigmaList[ind] for ind in range(17,len(sigmaList))]
#     else:
#         sigmaListOdd = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==1]
#     flag = 0
#     if (labels is None):
#         labels = ['' for init in range(0,inits)]
#         flag = 1
#     for init in range(0,inits):
#         axes.set_title(title)
#         axes.scatter(np.log(sigmaList[1:]), testLl[init,1:], color=colormap[color])
#         axes.plot(np.log(sigmaList[1:]), testLl[init,1:], color=colormap[color])
#         if(sigmaList[0]==0):
#             axes.scatter(-1 + np.log(sigmaList[1]), testLl[init,0], color=colormap[color], label=labels[init])
#             if (K==1):
#                 axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM'] + [f'{np.round(sigma,3)}' for sigma in sigmaListOdd])
#             else:
#                 axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM-HMM'] + [f'{np.round(sigma,3)}' for sigma in sigmaListOdd])
#         else:
#             axes.scatter(np.log(sigmaList[0]), testLl[init,0], color=colormap[color], label=f'init {init}')
#             axes.set_xticks([np.log(sigmaListEven)],[f'{np.round(sigma,2)}' for sigma in sigmaListEven])
#     axes.set_ylabel("Test LL (per trial)")
#     axes.set_xlabel("sigma")
#     if (flag == 0):
#         axes.legend()

#     if(save_fig==True):
#         plt.savefig(f'../figures/Sigma_vs_TestLl-{title}.png', bbox_inches='tight', dpi=400)


    