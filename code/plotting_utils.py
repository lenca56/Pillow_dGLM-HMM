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

def plotting_weights(w,sessInd, title):
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

    plt.plot(range(1,sess+1),w[sessInd[:-1],0,1,0],color='blue',marker='o',label='state 0 sensory')
    plt.plot(range(1,sess+1),w[sessInd[:-1],0,0,0],color='dodgerblue',marker='o', label='state 0 bias')
    plt.plot(range(1,sess+1),w[sessInd[:-1],1,1,0],color='darkred',marker='o',label='state 1 sensory')
    plt.plot(range(1,sess+1),w[sessInd[:-1],1,0,0],color='indianred',marker='o',label='state 1 bias')
    plt.xticks(range(1,sess+1))
    plt.ylabel("weights")
    plt.title(title)
    plt.xlabel('session')
    plt.legend()
    #plt.legend(fontsize='xx-small')
    plt.show()
    