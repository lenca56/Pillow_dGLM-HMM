# importing packages and modules
import pandas as pd 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from utils import *
from scipy.stats import multivariate_normal

class dGLM_HMM1():
    """
    Class for fitting driftinig GLM-HMM model 1 in which weights are constant within session but vary across sessions
    Code just works for c=2 at the moment
    Weights for class c=1 are always kept to 0 (so then emission probability becomes 1/(1+exp(-wTx)))

    Notation: 
        n: number of data points
        k: number of states (states)
        d: number of features (inputs to design matrix)
        c: number of classes (possible observations)
        X: design matrix (n x d)
        Y: observations (n x c) or (n x 1)
        w: weights mapping x to y (n x k x d x c)
    """

    def __init__(self, n, k, d, c):
            self.n, self.k, self.d, self.c  = n, k, d, c
    
    # Iris' code has reversed columns
    def emission_probability(self, x, w):
        '''
        Calculating emission probabilities for given design matrix x and weight matrix w

        Parameters
        ----------
        x: Ncurrent x D numpy array
        w: Ncurrent x K x D x C numpy array
            weight array

        Returns
        -------
        phi: Ncurrent x K x C numpy array
            emission probabilities
        '''
        
        Ncurrent = x.shape[0]

        phi = np.empty((Ncurrent, self.k, self.c)) # probability that it is state 1
        for k in range(0, self.k):
            for c in range(0, self.c):
                phi[:,k,c] = np.exp(-np.sum(w[:,k,:,c]*x,axis=1))
            phi[:,k,:]  = np.divide((phi[:,k,:]).T,np.sum(phi[:,k,:],axis=1)).T     

        return phi
    
    def simulate_data(self, trueW, trueP, sessInd, pi0=0.5):
        '''
        function that simulates X and Y data from true weights and true transition matrix
        S sessions, K states, D features (in the order: bias, sensory)
        C = 2 BINOMIAL

        Parameters
        ----------
        trueW: n x k x d x c numpy array
            true weight matrix. for c=2, trueW[:,:,:,1] = 0 
        trueP: k x k numpy array
            true probability transition matrix
        priorZstart: int
            0.5 probability of starting a session with state 0 (works for C=2)
        sessInd: list of int
            indices of each session start, together with last session end
            
        Returns
        -------
        x: n x d
            simulated design matrix 
        y: n x 1
            simulated observation matrix 
        z: n x 1
            simulated hidden states

        '''
        # check that weight and transition matrices are valid options
        if (trueW.shape != (self.n, self.k, self.d, self.c)):
            raise Exception(f'Weights need to have shape ({self.n}, {self.k}, {self.d}, {self.c})')
        
        if (trueP.shape != (self.k, self.k)):
            raise Exception(f'Transition matrix needs to have shape ({self.k}, {self.k})')
        
        x = np.empty((self.n, self.d))
        y = np.zeros((self.n, self.c)).astype(int)
        z = np.empty((self.n,),dtype=int)

        # input data x
        x[:,0] = 1 # bias term
        x[:,1] = stats.uniform.rvs(loc=-16,scale=33,size=self.n).astype(int)
        # standardizing sensory info
        x[:,1] = x[:,1] - x[:,1].mean()
        x[:,1] = x[:,1] / x[:,1].std()

        # latent variables z 
        for t in range(0, self.n):
            if (t in sessInd[:-1]): # beginning of session has a new draw for latent
                z[t] = np.random.binomial(n=1,p=1-pi0)
            else:
                z[t] = np.random.binomial(n=1, p=trueP[z[t-1],1])
        
        # observation probabilities
        phi = self.emission_probability(x, trueW)

        for t in range(0, self.n):
            y[t,int(np.random.binomial(n=1,p=phi[t,z[t],1]))]=1
        
        y = reshapeObs(y) # reshaping from n x c to n x 1

        return x, y, z
    
