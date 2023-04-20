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
from sklearn.model_selection import KFold

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

    oneSessInd = [0,N] # treating whole dataset as one session for normal GLM-HMM fitting
 
    for init in range(0,inits):
        for indSigma in range(0,len(sigmaList)): 
            print(indSigma)
            if (indSigma == 0): 
                if(sigmaList[0] == 0):
                    initP0, initW0 = dGLM_HMM.generate_param(sessInd=sessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]) 
                    allP[init, indSigma],  allW[init, indSigma], allLl[init, indSigma] = dGLM_HMM.fit(simX, simY,  initP0, initW0, sigma=reshapeSigma(1, K, D), sessInd=oneSessInd, pi0=None, maxIter=300, tol=1e-4) # sigma does not matter here
                else:
                    initP, initW = dGLM_HMM.generate_param(sessInd=sessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]) # initialize the model parameters
            else:
                initP = allP[init, indSigma-1] 
                initW = allW[init, indSigma-1] 
            
            if(sigmaList[indSigma] != 0):
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

def split_data_per_session(x, y, sessInd, folds=10, random_state=1):
    ''' 
    splitting data function for cross-validation, splitting for each session into folds and then merging
    currently does not balance number of trials for each session

    Parameters
    ----------
    x: n x d numpy array
        full design matrix
        y : n x 1 numpy vector 
            full vector of observations with values 0,1,..,C-1
        sessInd: list of int
            indices of each session start, together with last session end + 1

        Returns
        -------
        trainX: folds x train_size x d numpy array
            trainX[i] has train data of i-th fold
        trainY: folds x train_size  numpy array
            trainY[i] has train data of i-th fold
        trainSessInd: list of lists
            trainSessInd[i] have session start indices for the i-th fold of the train data
        testX: folds x test_size x d numpy array
            testX[i] has test data of i-th fold
        testY: folds x test_size  numpy array
            testY[i] has test data of i-th fold
        testSessInd: list of lists
            testSessInd[i] have session start indices for the i-th fold of the test data
        '''
    numberSessions = len(sessInd) - 1 # total number of sessions
    D = x.shape[1]
    N = x.shape[1]

    # initializing test and train size based on number of folds
    train_size = int(N - N/folds)
    test_size = int(N/folds)

    # initializing input and output arrays for each folds
    trainX = [[] for i in range(0,folds)]
    testX = [[] for i in range(0,folds)]
    trainY = [[] for i in range(0,folds)]
    testY = [[] for i in range(0,folds)]
    # initializing session indices for each fold
    trainSessInd = [[0] for i in range(0, folds)]
    testSessInd = [[0] for i in range(0, folds)]

    # splitting data for each fold for each session
    for sess in range(0,numberSessions):
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        for i, (train_index, test_index) in enumerate(kf.split(y[sessInd[sess]:sessInd[sess+1]])):
            trainSessInd[i].append(trainSessInd[i][-1] + len(train_index))
            testSessInd[i].append(testSessInd[i][-1] + len(test_index))
            trainY[i].append(y[sessInd[sess] + train_index])
            testY[i].append(y[sessInd[sess] + test_index])
            trainX[i].append(x[sessInd[sess] + train_index])
            testX[i].append(x[sessInd[sess] + test_index])
    
    array_trainX =  [np.zeros((trainSessInd[i][-1],D)) for i in range(0,folds)]
    array_testX = [np.zeros((testSessInd[i][-1],D)) for i in range(0,folds)]
    array_trainY = [np.zeros((trainSessInd[i][-1])) for i in range(0,folds)]
    array_testY = [np.zeros((testSessInd[i][-1])) for i in range(0,folds)]

    for sess in range(0,numberSessions):
        for i in range(0,folds):
            array_trainX[i][trainSessInd[i][sess]:trainSessInd[i][sess+1],:] = trainX[i][sess]
            array_testX[i][testSessInd[i][sess]:testSessInd[i][sess+1],:] = testX[i][sess]
            array_trainY[i][trainSessInd[i][sess]:trainSessInd[i][sess+1]] = trainY[i][sess]
            array_testY[i][testSessInd[i][sess]:testSessInd[i][sess+1]] = testY[i][sess]
            
    return array_trainX, array_trainY, trainSessInd, array_testX, array_testY, testSessInd

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
