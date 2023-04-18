# importing packages and modules
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from utils import *
from plotting_utils import *
import dglm_hmm1
from scipy.stats import multivariate_normal, norm
import seaborn as sns

def fit_multiple_sigmas(N,K,D,C, sessInd, sigmaList=[0.01,0.032,0.1,0.32,1,10,100], inits=1, maxiter=400, modelType='drift', save=False):
    ''' 
    fitting function for multiple values of sigma with initializing from the previously found parameters with increasing order of fitting sigma
    '''

    dGLM_HMM = dglm_hmm1.dGLM_HMM1(N,K,D,C)
    simX = np.load(f'../data/N={N}_{K}_state_{modelType}_trainX.npy')
    simY = np.load(f'../data/N={N}_{K}_state_{modelType}_trainY.npy')

    allLl = np.zeros((inits, len(sigmaList), maxiter))
    allP = np.zeros((inits, len(sigmaList), K,K))
    allW = np.zeros((inits, len(sigmaList),N,K,D,C))
 
    for init in range(0,inits):
        for indSigma in range(0,len(sigmaList)): 
            print(indSigma)
            if (indSigma == 0): 
                initP, initW = dGLM_HMM.generate_param(sessInd=sessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]) # initialize the model parameters
            else:
                initP = allP[init, indSigma-1] 
                initW = allW[init, indSigma-1] 
            
            print(indSigma)
            # fit on whole dataset
            allP[init, indSigma],  allW[init, indSigma], allLl[init, indSigma] = dGLM_HMM.fit(simX, simY,  initP, initW, sigma=reshapeSigma(sigmaList[indSigma], K, D), sessInd=sessInd, pi0=None, maxIter=maxiter, tol=1e-3) # fit the model
                
    if(save==True):
        np.save(f'../data/Ll_N={N}_{K}_state_{modelType}', allLl)
        np.save(f'../data/P_N={N}_{K}_state_{modelType}', allP)
        np.save(f'../data/W_N={N}_{K}_state_{modelType}', allW)

    return allLl, allP, allW

def evaluate_multiple_sigmas(N,K,D,C, trainSessInd=None, testSessInd=None, sigmaList=[0.01,0.032,0.1,0.32,1,10,100], modelType='drift', save=False):
    ''' 
    function for evaluating previously fit models with different sigmas on the same test set
    '''
    testX = np.load(f'../data/{K}_state_{modelType}_testX.npy')
    testY = np.load(f'../data/{K}_state_{modelType}_testY.npy')
    allP = np.load(f'../data/P_N={N}_{K}_state_{modelType}.npy')
    allW = np.load(f'../data/W_N={N}_{K}_state_{modelType}.npy')
    sess = 10
    inits = allW.shape[0]
    testLl = np.zeros((inits, len(sigmaList)))
    dGLM_HMM = dglm_hmm1.dGLM_HMM1(N,K,D,C)

    if (trainSessInd is None): # same length sessions
        trainT = int(N/sess)
        trainSessInd = [x for x in range(0,trainT*sess+trainT, trainT)]
    if (testSessInd is None): # same length sessions
        testT = int(testX.shape[0]/sess)
        testSessInd = [x for x in range(0,testT*sess+testT,testT)]
    
    # Evaluate on test data
    for init in range(0,inits):
        for indSigma in range(0, len(sigmaList)):  
            for s in range(0, sess):
                # evaluate on test data for each session separately
                testPhi = dGLM_HMM.observation_probability(testX, reshapeWeights(allW[init, indSigma], trainSessInd, testSessInd))
                _, _, temp = dGLM_HMM.forward_pass(testY[testSessInd[s]:testSessInd[s+1]],allP[init, indSigma],testPhi[testSessInd[s]:testSessInd[s+1]])
                testLl[init, indSigma] += temp
    
    testLl = testLl / testSessInd[-1] # normalizing to the total number of trials in test

    if(save==True):
        np.save(f'../data/testtLl_N={N}_{K}_state_{modelType}', testLl)
    
    return testLl

def sigma_testLl_plot(sigmaList, testLl, axes, title='', label='', save_fig=False):
    ''' 
    function for plotting the test LL vs sigma scalars
    '''
    inits = testLl.shape[0]
    colormap = sns.color_palette("viridis")
    for init in range(0,inits):
        axes.set_title(title)
        #for indSigma in range(0, len(sigmaList)):
        axes.scatter(np.log(sigmaList[:]), testLl[init,:], color=colormap[init])
        axes.set_ylabel("Test LL (per trial)")
        axes.set_xticks(np.log(sigmaList),[f'log({sigma})' for sigma in sigmaList])
        axes.set_xlabel("Log(sigma)")
        axes.scatter(np.log(sigmaList[-1]), testLl[init,-1], color=colormap[init],label=f'init {init} - {label}')
        axes.legend()

    if(save_fig==True):
        plt.savefig(f'../figures/{title}', bbox_inches='tight', dpi=300)

def accuracy(x,y,z,s):
    '''
    Calculates and plots percentage accuracy (given X and Y) and percentage accuracy in state 0 (given Z)

    X: N x D numpy array
    Y: N x 1 numpy array
    s: int
        number of sessions
    '''
    n = x.shape[0]
    perf = np.zeros((s,))
    trials = int(n/s)
    state0 = np.empty((s,))
    ind = []
    for sess in range(0,s):
        state0[sess] = trials - z[sess*trials:(sess+1)*trials].sum()
        for t in range(0,trials):
            if (x[sess*trials+t,1]>0 and y[sess*trials+t,1]==1):
                perf[sess]+=1
            elif (x[sess*trials+t,1]<0 and y[sess*trials+t,0]==1):
                perf[sess]+=1
            else:
                ind.append(sess*trials+t)

    perf = perf / trials # normalize to number of trials per session
    state0 = state0 / trials
    plt.plot(range(0,s),state0*100,marker='o',color='darkgray',label='in state 0')
    plt.plot(range(0,s),perf*100,marker='o',label="accurracy")
    plt.ylabel("percentage %")
    plt.xlabel("sesion")
    plt.legend(fontsize='small')
    plt.show()

    return perf, ind
