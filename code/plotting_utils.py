import pandas as pd 
import numpy as np
from pathlib import Path
import math
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats
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
    