# importing packages and modules
import pandas as pd 
import numpy as np
from pathlib import Path
import math
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats

def reshapeObs(y):
    '''
    reshaping observation y from having C columns with values among 0 and 1 to having 1 column with values from 0 to C-1

    Parameters
    ----------

    Returns
    -------
    '''
    
    yNew = np.empty((y.shape[0],))
    if(y.shape[1] > 1):
        yNew = np.array(np.where(y==1)[1]).reshape(y.shape[0],)
    return yNew

def reshapeSigma(sigma, K, D):
    ''' 
    changing variance parameter sigma to have shape KxD

    Parameters
    ----------
    sigma: nonnegative floats
        either scalar, Kx1, 1xD, or KxD numpy array

    Returns
    -------
    newSigma: KxD numpy array
    '''
    newSigma = np.empty((K,D))
    if (isinstance(sigma, float)  == True) or isinstance(sigma, int)  == True:
        newSigma.fill(sigma)
    elif (sigma.shape[0]==1 and sigma.shape[1]==D):
        newSigma = np.repeat(sigma, repeats = K, axis=0)
    elif (sigma.shape[0]==K and sigma.shape[1]==1):
        newSigma = np.repeat(sigma, repeats = D, axis=1)
    elif (sigma.shape[0]==K and sigma.shape[1]==D):
        newSigma = np.copy(sigma)
    else:
        raise Exception('sigma can only be scalar, Kx1, 1xD, or KxD numpy array')
    
    # add check of nonnegative elements

    return newSigma

def reshapeWeights(w, oldSessInd, newSessInd):
    ''' 
    reshaping weights from session indices of oldSessInd to session indices of newSessInd

    Parameters
    ----------
    w: T x k x d x c numpy array
            true weight matrix. for c=2, trueW[:,:,:,1] = 0 
    oldSessInd: list of int
        old indices of each session start, together with last session end + 1
    newSessInd: list of int
        new indices of each session start, together with last session end + 1
            
    Returns
    -------
    reshapedW: newT x k x d x c
    '''
    T = w.shape[0]
    k = w.shape[1]
    d = w.shape[2]
    c = w.shape[3]
    if (T != oldSessInd[-1]):
        raise Exception ("Indices and weights do not match in size")
    if (len(oldSessInd) != len(newSessInd)):
        raise Exception ("old and new indices don't have the same number of sessions")
    
    newT = newSessInd[-1]
    reshapedW = np.zeros((newT, k, d, c))
    for sess in range(0,len(oldSessInd)-1):
        reshapedW[newSessInd[sess]:newSessInd[sess+1],:,:,0] = w[oldSessInd[sess],:,:,0]
    
    return reshapedW


def permute_states(w, sessInd):
    ''' 
    decreasing order acording drift of sensory across consecutive sessions
    '''
    k = w.shape[1]
    sess = len(sessInd)-1
    driftState = np.zeros((k,))
    for s in range(0,sess-1):
        for i in range(0,k):
            driftState[i]+= abs(w[sessInd[s+1],i,1,0] - w[sessInd[s],i,1,0]) # not sure about the scale
    sortedInd = list(np.argsort(driftState))
    sortedInd.reverse() # decreasing order
    
    return w[:,sortedInd,:,:]
    
