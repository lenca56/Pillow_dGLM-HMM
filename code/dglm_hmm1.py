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
    X columns represent [bias, sensory] in this order

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
    def observation_probability(self, x, w):
        '''
        Calculating observation probabilities for given design matrix x and weight matrix w

        Parameters
        ----------
        x: Ncurrent x D numpy array
            input matrix
        w: Ncurrent x K x D x C numpy array
            weight matrix

        Returns
        -------
        phi: Ncurrent x K x C numpy array
            observation probabilities matrix
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

        Parameters
        ----------
        trueW: n x k x d x c numpy array
            true weight matrix. for c=2, trueW[:,:,:,1] = 0 
        trueP: k x k numpy array
            true probability transition matrix
        priorZstart: int
            0.5 probability of starting a session with state 0 (works for C=2)
        sessInd: list of int
            indices of each session start, together with last session end + 1
        pi0: float
            constant between 0 and 1, representing probability that first latent in a session is state 0
            
        Returns
        -------
        x: n x d
            simulated design matrix
        y: n x 1
            simulated observation vector
        z: n x 1
            simulated hidden states vector

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
        phi = self.observation_probability(x, trueW)

        for t in range(0, self.n):
            y[t,int(np.random.binomial(n=1,p=phi[t,z[t],1]))]=1
        
        y = reshapeObs(y) # reshaping from n x c to n x 1

        return x, y, z

    # already checked with Iris' function that it is correct
    def forward_pass(self, y, P, phi, pi0=None):
        '''
        Calculates alpha scaled as part of the forward-backward algorithm in E-step 
       
        Parameters
        ----------
        y : T x 1 numpy vector 
            vector of observations with values 0,1,..,C-1
        P : k x k numpy array 
            matrix of transition probabilities
        phi : T x k x  c numpy array
            matrix of observation probabilities
        pi0: k x 1 numpy vector
            distribution of first state before it has sesn any data 
        Returns
        -------
        ll : float
            marginal log-likelihood of the data p(y)
        alpha : T x k numpy vector
            matrix of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        ct : T x 1 numpy vector
            vector of the forward marginal likelihoods p(y_t | y_1:t-1)
        '''
        T = y.shape[0]
        
        alpha = np.zeros((T, self.k)) # forward probabilities p(z_t | y_1:t)
        alpha_prior = np.zeros((T, self.k)) # prior probabilities p(z_t | y_1:t-1)
        lt = np.zeros((T, self.k)) # likelihood of data p(y_t|z_t)
        ct = np.zeros(T) # forward marginal likelihoods p(y_t | y_1:t-1)

        # forward pass calculations
        for t in range(0,T):
            lt[t,:] = phi[t,:,y[t]] # likelihood p(y_t | z_t)
            if (t==0): # time point 0
                # prior of z_0 before any data 
                if (pi0==None):
                    alpha_prior[0,:] = np.ones((1,self.k))/self.k # uniform prior
                else:
                    alpha_prior[0,:] = pi0
            else:
                alpha_prior[t,:] = (alpha[t-1,:].T @ P) # conditional p(z_t | y_1:t-1)
            pxz = np.multiply(lt[t],alpha_prior[t,:]) # joint P(y_1:t, z_t)
            ct[t] = np.sum(pxz) # conditional p(y_t | y_1:t-1)
            alpha[t,:] = pxz/ct[t] # conditional p(z_t | y_1:t)
        
        ll = np.sum(np.log(ct)) # marginal log likelihood p(y_1:T) as sum of log conditionals p(y_t | y_1:t-1) 
        
        return alpha, ct, ll
    
    # already checked with Iris' function that it is correct
    def backward_pass(self, y, P, phi, ct, pi0=None):
        '''
        Calculates beta scaled as part of the forward-backward algorithm in E-step 

        Parameters
        ----------
        y : T x 1 numpy vector
            vector of observations with values 0,1,..,C-1
        p : k x k numpy array
            matrix of transition probabilities
        phi : T x k x c numppy array
            matrix of observation probabilities
        ct : T x 1 numpy vector 
            veector of forward marginal likelihoods p(y_t | y_1:t-1), calculated at forward_pass
            
        Returns
        -------
        beta: T x k numpy array 
            matrix of backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)
        '''

        T = y.shape[0]
        
        beta = np.zeros((T, self.k)) # backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)
        lt = np.zeros((T, self.k)) # likelihood of data p(y_t|z_t)

        # last time point
        beta[-1] = 1 # p(z_T=1)

        # backward pass calculations
        for t in np.arange(T-2,-1,-1):
            lt[t+1,:] = phi[t+1,:,y[t+1]] 
            beta[t,:] = P @ (np.multiply(beta[t+1,:],lt[t+1,:]))
            beta[t,:] = beta[t,:] / ct[t+1] # scaling factor
        
        return beta
    
    def posteriorLatents(self, y, p, phi, alpha, beta, ct):
        ''' 
        calculates marginal posterior of latents gamma(z_t) = p(z_t | y_1:T)
        and joint posterior of successive latens zeta(z_t, z_t+1) = p(z_t, z_t+1 | y_1:T)

        Parameters
        ----------
        y : Tx1 numpy vector 
            vector of observations with values 0,1,..,C-1
        p : k x k numpy array
            matrix of transition probabilities
        phi : T x k x c numppy array
            matrix of observation probabilities
        alpha : T x k numpy vector
            marix of the conditional probabilities p(z_t | x_1:t, y_1:t)
        beta: T x k numpy array 
            matrix of backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)
        ct : T x 1 numpy vector
            vector of the forward marginal likelihoods p(y_t | y_1:t-1)
        
        Returns
        -------
        gamma: T x k numpy array
            matrix of marginal posterior of latents p(z_t | y_1:T)
        zeta: T-1 x k x k 
            matrix of joint posterior of successive latens p(z_t, z_t+1 | y_1:T)
        '''
        
        T = ct.shape[0]
        gamma = np.empty((T, self.k)).astype(float) # marginal posterior of latents
        zeta = np.empty((T-1, self.k, self.k)).astype(float) # joint posterior of successive latents

        gamma = np.multiply(alpha, beta) # gamma(z_t) = alpha(z_t) * beta(z_t)

        # zeta(z_t, z_t+1) =  alpha(z_t) * beta(z_t+1) * p (z_t+1 | z_t) * p(y_t+1 | z_t+1) / c_t+1
        for t in range(0,T-1):
            alpha_beta = alpha[t,:].reshape((self.k, 1)) @ beta[t+1,:].reshape((1, self.k))
            zeta[t,:,:] = np.multiply(alpha_beta,p) 
            zeta[t,:,:] = np.multiply(zeta[t,:,:],phi[t+1,:,y[t+1]]) # change t+1 to t in phi to match Iris'
            zeta[t,:,:] = zeta[t,:,:] / ct[t+1]
            
        return gamma, zeta

    
