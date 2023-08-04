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
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '..', 'LC_PWM_GLM-HMM/code')))
import io_utils, analysis_utils, plotting_utils
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def split_data(x, y, sessInd, folds=4, blocks=10, random_state=1):
    ''' 
    splitting data function for cross-validation
    currently does not balance trials for each session

    !Warning: each session must have at least (folds-1) * blocks  trials

    Parameters
    ----------
    x: n x d numpy array
        full design matrix
    y : n x 1 numpy vector 
        full vector of observations with values 0,1,..,C-1
    sessInd: list of int
        indices of each session start, together with last session end + 1
    folds: int
        number of folds to split the data in (test has 1/folds points of whole dataset)
    blocks: int (default = 10)
        blocks of trials to keep together when splitting data (to keep some time dependencies)
    random_state: int (default=1)
        random seed to always get the same split if unchanged

    Returns
    -------
    trainX: list of train_size[i] x d numpy arrays
        trainX[i] has input train data of i-th fold
    trainY: list of train_size[i] numpy arrays
        trainY[i] has output train data of i-th fold
    trainSessInd: list of lists
        trainSessInd[i] have session start indices for the i-th fold of the train data
    testX: // same for test
    '''
    N = x.shape[0]
    D = x.shape[1]
    
    # initializing session indices for each fold
    trainSessInd = [[0] for i in range(0, folds)]
    testSessInd = [[0] for i in range(0, folds)]

    # split session indices into blocks and get session indices for train and test
    totalSess = len(sessInd) - 1
    for s in range(0, totalSess):
        ySessBlock = np.arange(0, (sessInd[s+1]-sessInd[s])/blocks)
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state) # shuffle=True and random_state=int for random splitting, otherwise it's consecutive
        for i, (train_blocks, test_blocks) in enumerate(kf.split(ySessBlock)):
            train_indices = []
            test_indices = []
            for b in ySessBlock:
                if (b in train_blocks):
                    train_indices = train_indices + list(np.arange(sessInd[s] + b*blocks, min(sessInd[s] + (b+1) * blocks, sessInd[s+1])))
                elif(b in test_blocks):
                    test_indices = test_indices + list(np.arange(sessInd[s] + b*blocks, min(sessInd[s] + (b+1) * blocks, sessInd[s+1])))
                else:
                    raise Exception("Something wrong with session block splitting")

            trainSessInd[i].append(len(train_indices)+ trainSessInd[i][-1])
            testSessInd[i].append(len(test_indices) + testSessInd[i][-1])

    # initializing input and output arrays for each folds
    trainX = [np.zeros((trainSessInd[i][-1], D)) for i in range(0,folds)]
    trainY = [np.zeros((trainSessInd[i][-1])).astype(int) for i in range(0,folds)]
    testX = [np.zeros((testSessInd[i][-1], D)) for i in range(0,folds)]
    testY = [np.zeros((testSessInd[i][-1])).astype(int) for i in range(0,folds)]

    # same split as above but now get the actual data split
    for s in range(0, totalSess):
        ySessBlock = np.arange(0, (sessInd[s+1]-sessInd[s])/blocks)
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state) # shuffle=True and random_state=int for random splitting, otherwise it's consecutive
        for i, (train_blocks, test_blocks) in enumerate(kf.split(ySessBlock)):
            train_indices = []
            test_indices = []
            for b in ySessBlock:
                if (b in train_blocks):
                    train_indices = train_indices + list(np.arange(sessInd[s] + b*blocks, min(sessInd[s] + (b+1) * blocks, sessInd[s+1])))
                elif(b in test_blocks):
                    test_indices = test_indices + list(np.arange(sessInd[s] + b*blocks, min(sessInd[s] + (b+1) * blocks, sessInd[s+1])))
                else:
                    raise Exception("Something wrong with session block splitting")
            trainX[i][trainSessInd[i][s]:trainSessInd[i][s+1]] = x[np.array(train_indices).astype(int)]
            trainY[i][trainSessInd[i][s]:trainSessInd[i][s+1]] = y[np.array(train_indices).astype(int)]
            testX[i][testSessInd[i][s]:testSessInd[i][s+1]] = x[np.array(test_indices).astype(int)]
            testY[i][testSessInd[i][s]:testSessInd[i][s+1]] = y[np.array(test_indices).astype(int)]

    return trainX, trainY, trainSessInd, testX, testY, testSessInd


def fit_multiple_sigmas_simulated(N,K,D,C, sessInd, sigmaList=[0.01,0.032,0.1,0.32,1,10,100], inits=1, maxiter=400, modelType='drift', save=False):
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
                    initP0, initW0 = dGLM_HMM.generate_param(sessInd=oneSessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]) 
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

def fit_eval_CV_multiple_sigmas(x, y, sessInd, K, splitFolds, fitFolds=1, sigmaList=[0, 0.01, 0.1, 1, 10, 100], maxiter=300, glmhmmW=None, glmhmmP=None, L2penaltyW=1, priorDirP = [10,1]):
    ''' 
    fitting function for multiple values of sigma with initializing from the previously found parameters with increasing order of fitting sigma
    first sigma is 0 and is the GLM-HMM fit
    takes arguments as CV data

    Parameters
    ----------
    x: n x d numpy array
        full design matrix
    y : n x 1 numpy vector 
        full vector of observations with values 0,1,..,C-1
    sessInd: list of int
        indices of each session start, together with last session end + 1
    K: int
        number of latent states
    splitFolds: int
        number of folds to split the data in (test has approx 1/folds points of whole dataset)
    fitFolds: int 
        number of folds to actually train on (default=1)
    sigmaList: list of positive numbers, starting with 0 (default =[0, 0.01, 0.1, 1, 10, 100])
        weight drifting hyperparameter list
    maxiter: int 
        maximum number of iterations before EM stopped (default=300)
    glmhmmW: Nsize x K x D x C numpy array 
        given weights from glm-hmm fit (default=None)
    glmhmmP=None: K x K numpy array 
        given transition matrix from glm-hmm fit (default=None)
    L2penaltyW: int
        positive value determinig strength of L2 penalty on weights when fitting (default=1)
    priorDirP : list of length 2
        first number is Dirichlet prior on diagonal, second number is the off-diagonal (default = [10,1])

    Returns
    ----------
    trainLl: list of length fitFolds 
        trainLl[i] is len(sigmaList) x maxiter numpy array of training log like for i'th fold
    testLl: list of length fitFolds 
        testLl[i] is len(sigmaList) numpy vector of normalized test log like for i'th fold
    allP: list of length fitFolds 
        allP[i] if len(sigmaList) x K x K numpy array of fit transition matrix for i'th fold
    allW: list of length fitFolds 
    '''

    # splitting data into splitFolds number of folds - each session is split individually with blocks of 10 kept together
    trainX, trainY, trainSessInd, testX, testY, testSessInd = split_data(x, y, sessInd, folds=splitFolds, blocks=10, random_state=1)
    D = trainX[0].shape[1]
    C = 2 # only looking at binomial classes

    trainLl = [np.zeros((len(sigmaList), maxiter)) for i in range(0, fitFolds)] 
    testLl = [np.zeros((len(sigmaList))) for i in range(0, fitFolds)]
    allP = [np.zeros((len(sigmaList), K, K)) for i in range(0, fitFolds)] 
    allW = [] 

    for fold in range(0, fitFolds):
        # initializing parameters for each fold
        N = trainX[fold].shape[0]
        print(f'Fold {fold} training size {N}')
        oneSessInd = [0,N] # treating whole dataset as one session for normal GLM-HMM fitting
        dGLM_HMM = dglm_hmm1.dGLM_HMM1(N,K,D,C)
        allW.append(np.zeros((len(sigmaList), N,K,D,C)))
        trainY[fold] = trainY[fold].astype(int)
        testY[fold] = testY[fold].astype(int)

        for indSigma in range(0,len(sigmaList)): 
            print("Sigma Index " + str(indSigma))
            if (indSigma == 0): 
                if(sigmaList[0] == 0): 
                    if (glmhmmW is not None and glmhmmP is not None): # if parameters are given from standard GLM-HMM 
                        print("GLM HMM GIVEN INIT")
                        oldSessInd = [0, glmhmmW.shape[0]] # assuming glmhmmW has constant weights
                        allP[fold][indSigma] = np.copy(glmhmmP) # K x K transition matrix
                        allW[fold][indSigma] = reshapeWeights(glmhmmW, oldSessInd, oneSessInd, standardGLMHMM=True)
                    else:
                        initP0, initW0 = dGLM_HMM.generate_param(sessInd=oneSessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]) 
                        allP[fold][indSigma],  allW[fold][indSigma], trainLl[fold][indSigma] = dGLM_HMM.fit(trainX[fold], trainY[fold],  initP0, initW0, sigma=reshapeSigma(1, K, D), sessInd=oneSessInd, pi0=None, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW, priorDirP=priorDirP) # sigma does not matter here
                
                else:
                    raise Exception("First sigma of sigmaList should be 0, meaning standard GLM-HMM")
            else:
                # initializing from previous fit
                initP = allP[fold][indSigma-1] 
                initW = allW[fold][indSigma-1] 
            
                # fitting dGLM-HMM
                allP[fold][indSigma],  allW[fold][indSigma], trainLl[fold][indSigma] = dGLM_HMM.fit(trainX[fold], trainY[fold],  initP, initW, sigma=reshapeSigma(sigmaList[indSigma], K, D), sessInd=trainSessInd[fold], pi0=None, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW, priorDirP=priorDirP) # fit the model
        
            # evaluate on the test set
            sess = len(trainSessInd[fold]) - 1 # number sessions
            testPhi = dGLM_HMM.observation_probability(testX[fold], reshapeWeights(allW[fold][indSigma], trainSessInd[fold], testSessInd[fold]))
            for s in range(0, sess):
                # evaluate on test data for each session separately
                _, _, temp = dGLM_HMM.forward_pass(testY[fold][testSessInd[fold][s]:testSessInd[fold][s+1]],allP[fold][indSigma],testPhi[testSessInd[fold][s]:testSessInd[fold][s+1]])
                testLl[fold][indSigma] += temp
    
        testLl[fold] = testLl[fold] / testSessInd[fold][-1] # normalizing to the total number of trials in test dataset

    return trainLl, testLl, allP, allW, trainSessInd, testSessInd

# OLD FUNCTION - REPLACED BY ABOVE ONE

# def fit_eval_CV_multiple_sigmas_PWM(rat_id, stage_filter, K, folds=3, sigmaList=[0, 0.01, 0.1, 1, 10, 100], maxiter=300, glmhmmW=None, glmhmmP=None, L2penaltyW=1, path=None, save=False):
#     ''' 
#     fitting function for multiple values of sigma with initializing from the previously found parameters with increasing order of fitting sigma
#     first sigma is 0 and is the GLM-HMM fit
#     only suited for PWM data for now
#     '''
#     x, y = io_utils.prepare_design_matrices(rat_id=rat_id, path=path, psychometric=True, cutoff=10, stage_filter=stage_filter, overwrite=False)
#     sessInd = list(io_utils.session_start(rat_id=rat_id, path=path, psychometric=True, cutoff=10, stage_filter=stage_filter)) 
#     trainX, trainY, trainSessInd, testX, testY, testSessInd = split_data_per_session(x, y, sessInd, folds=folds, random_state=1)
#     D = trainX[0].shape[1]
#     C = 2 # only looking at binomial classes

#     trainLl = [np.zeros((len(sigmaList), maxiter)) for i in range(0,folds)] 
#     testLl = [np.zeros((len(sigmaList))) for i in range(0,folds)]
#     allP = [np.zeros((len(sigmaList), K,K)) for i in range(0,folds)] 
#     allW = [] 

#     for fold in [0]: # fitting single fold     # fittinng all folds -> range(0,folds): 
#         # initializing parameters for each fold
#         N = trainX[fold].shape[0]
#         oneSessInd = [0,N] # treating whole dataset as one session for normal GLM-HMM fitting
#         dGLM_HMM = dglm_hmm1.dGLM_HMM1(N,K,D,C)
#         allW.append(np.zeros((len(sigmaList), N,K,D,C)))
#         trainY[fold] = trainY[fold].astype(int)
#         testY[fold] = testY[fold].astype(int)

#         for indSigma in range(0,len(sigmaList)): 
#             print("Sigma Index " + str(indSigma))
#             if (indSigma == 0): 
#                 if(sigmaList[0] == 0):
#                     if (glmhmmW is not None and glmhmmP is not None):
#                         # best found glmhmm with multiple initializations - constant P and W
#                         print("GLM HMM BEST INIT")
#                         oldSessInd = [0, glmhmmW.shape[0]]
#                         initP0 = np.copy(glmhmmP) # K x K transition matrix
#                         initW0 = reshapeWeights(glmhmmW, oldSessInd, oneSessInd, standardGLMHMM=True)
#                     else:
#                         initP0, initW0 = dGLM_HMM.generate_param(sessInd=oneSessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]) 
#                     # fitting for sigma = 0
#                     allP[fold][indSigma],  allW[fold][indSigma], trainLl[fold][indSigma] = dGLM_HMM.fit(trainX[fold], trainY[fold],  initP0, initW0, sigma=reshapeSigma(1, K, D), sessInd=oneSessInd, pi0=None, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW) # sigma does not matter here
#                 else:
#                     raise Exception("First sigma of sigmaList should be 0, meaning standard GLM-HMM")
#             else:
#                 # initializing from previoous fit
#                 initP = allP[fold][indSigma-1] 
#                 initW = allW[fold][indSigma-1] 
            
#                 # fitting dGLM-HMM
#                 allP[fold][indSigma],  allW[fold][indSigma], trainLl[fold][indSigma] = dGLM_HMM.fit(trainX[fold], trainY[fold],  initP, initW, sigma=reshapeSigma(sigmaList[indSigma], K, D), sessInd=trainSessInd[fold], pi0=None, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW) # fit the model
        
#             # evaluate
#             sess = len(trainSessInd[fold]) - 1 # number sessions
#             testPhi = dGLM_HMM.observation_probability(testX[fold], reshapeWeights(allW[fold][indSigma], trainSessInd[fold], testSessInd[fold]))
#             for s in range(0, sess):
#                 # evaluate on test data for each session separately
#                 _, _, temp = dGLM_HMM.forward_pass(testY[fold][testSessInd[fold][s]:testSessInd[fold][s+1]],allP[fold][indSigma],testPhi[testSessInd[fold][s]:testSessInd[fold][s+1]])
#                 testLl[fold][indSigma] += temp
    
#         testLl[fold] = testLl[fold] / testSessInd[fold][-1] # normalizing to the total number of trials in test dataset

#         if(save==True):
#             np.save(f'../data_PWM/trainLl_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', trainLl[fold])
#             np.save(f'../data_PWM/testLl_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', testLl[fold])
#             np.save(f'../data_PWM/P_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', allP[fold])
#             np.save(f'../data_PWM/W_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', allW[fold])
#             np.save(f'../data_PWM/trainSessInd_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', np.array(trainSessInd[fold]))
#             np.save(f'../data_PWM/testSessInd_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', np.array(testSessInd[fold]))

#     return trainLl, testLl, allP, allW

def evaluate_multiple_sigmas_simulated(N,K,D,C, trainSessInd=None, testSessInd=None, sigmaList=[0.01,0.032,0.1,0.32,1,10,100], modelType='drift', save=False):
    ''' 
    function for evaluating previously fit models with different sigmas on the same test set of simulated data
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
        np.save(f'../data/testLl_N={N}_{K}_state_{modelType}', testLl)
    
    return testLl

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

def find_top_init_plot_loglikelihoods(ll,maxdiff,ax=None,startix=5,plot=True):
    '''
    Function from Iris' GLM-HMM github with some alterations
    
    Plot the trajectory of the log-likelihoods for multiple fits, identify how many top fits (nearly) match, and 
    color those trajectories in the plot accordingly

    Parameters
    ----------

    Returns
    ----------

    '''

    # replacing the 0's by nan's
    lls = np.copy(ll)
    lls[lls==0] = np.nan

    # get the final ll for each fit
    final_lls = np.array([np.amax(lls[i,~np.isnan(lls[i,:])]) for i in range(lls.shape[0])])
    
    # get the index of the top ll
    bestInd = np.argmax(final_lls)
    
    # compute the difference between the top ll and all final lls
    ll_diffs = final_lls[bestInd] - final_lls
    
    # identify te fits where the difference from the top ll is less than maxdiff
    top_matching_lls = lls[ll_diffs < maxdiff,:]
    
    # plot
    if (plot == True):
        ax.plot(np.arange(startix,lls.shape[1]),lls.T[startix:], color='black')
        ax.plot(top_matching_lls.T[startix:], color='red')
        ax.set_xlabel('iterations of EM', fontsize=16)
        ax.set_ylabel('log-likelihood', fontsize=16)
    
    return bestInd, final_lls, np.where(ll_diffs < maxdiff)[0] # return indices of best (matching) fits

def get_mouse_design(dfAll, subject, sessStop=-1):
    ''' 
    function to give design matrix x and output vector y for a given subject until session sessStpo
    '''
    data = dfAll[dfAll['subject']==subject]   # Restrict data to the subject specified

    p=5 # as used in Psytrack paper
    data['cL'] = np.tanh(p*data['contrastLeft'])/np.tanh(p) # tanh transformation of left contrasts
    data['cR'] = np.tanh(p*data['contrastRight'])/np.tanh(p) # tanh transformation of right contrasts

    # keeping first 40 sessions
    dateToKeep = np.unique(data['date'])[0:sessStop]
    dataTemp = pd.DataFrame(data.loc[data['date'].isin(list(dateToKeep))])

    # design and out matrix
    x = np.ones((dataTemp.shape[0], 3)) # column 0 is bias
    x[:,1] = dataTemp['cL'] # cL = contrast left transformed 
    x[:,2] = dataTemp['cR'] # cR = contrast right transformed
    y = np.array(dataTemp['choice'])
    print(y.shape)

    # session start indicies
    sessInd = [0]
    for date in dateToKeep :
        d = dataTemp[dataTemp['date']==date]
        for sess in np.unique(d['session']):
            dTemp = d[d['session'] == sess] 
            dLength = len(dTemp.index.tolist())
            sessInd.append(sessInd[-1] + dLength)
    print(sessInd[-1])
    
    return x, y, sessInd

# OLD SPLITTING DATA FUNCTION

# def split_data_per_session(x, y, sessInd, folds=10, random_state=1):
#     ''' 
#     splitting data function for cross-validation, splitting for each session into folds and then merging
#     currently does not balance number of trials for each session

#     Parameters
#     ----------
#     x: n x d numpy array
#         full design matrix
#         y : n x 1 numpy vector 
#             full vector of observations with values 0,1,..,C-1
#         sessInd: list of int
#             indices of each session start, together with last session end + 1

#         Returns
#         -------
#         trainX: folds x train_size x d numpy array
#             trainX[i] has train data of i-th fold
#         trainY: folds x train_size  numpy array
#             trainY[i] has train data of i-th fold
#         trainSessInd: list of lists
#             trainSessInd[i] have session start indices for the i-th fold of the train data
#         testX: folds x test_size x d numpy array
#             testX[i] has test data of i-th fold
#         testY: folds x test_size  numpy array
#             testY[i] has test data of i-th fold
#         testSessInd: list of lists
#             testSessInd[i] have session start indices for the i-th fold of the test data
#         '''
#     numberSessions = len(sessInd) - 1 # total number of sessions
#     D = x.shape[1]
#     N = x.shape[1]

#     # initializing test and train size based on number of folds
#     train_size = int(N - N/folds)
#     test_size = int(N/folds)

#     # initializing input and output arrays for each folds
#     trainX = [[] for i in range(0,folds)]
#     testX = [[] for i in range(0,folds)]
#     trainY = [[] for i in range(0,folds)]
#     testY = [[] for i in range(0,folds)]
#     # initializing session indices for each fold
#     trainSessInd = [[0] for i in range(0, folds)]
#     testSessInd = [[0] for i in range(0, folds)]

#     # splitting data for each fold for each session
#     for sess in range(0,numberSessions):
#         kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
#         for i, (train_index, test_index) in enumerate(kf.split(y[sessInd[sess]:sessInd[sess+1]])):
#             trainSessInd[i].append(trainSessInd[i][-1] + len(train_index))
#             testSessInd[i].append(testSessInd[i][-1] + len(test_index))
#             trainY[i].append(y[sessInd[sess] + train_index])
#             testY[i].append(y[sessInd[sess] + test_index])
#             trainX[i].append(x[sessInd[sess] + train_index])
#             testX[i].append(x[sessInd[sess] + test_index])
    
#     array_trainX =  [np.zeros((trainSessInd[i][-1],D)) for i in range(0,folds)]
#     array_testX = [np.zeros((testSessInd[i][-1],D)) for i in range(0,folds)]
#     array_trainY = [np.zeros((trainSessInd[i][-1])) for i in range(0,folds)]
#     array_testY = [np.zeros((testSessInd[i][-1])) for i in range(0,folds)]

#     for sess in range(0,numberSessions):
#         for i in range(0,folds):
#             array_trainX[i][trainSessInd[i][sess]:trainSessInd[i][sess+1],:] = trainX[i][sess]
#             array_testX[i][testSessInd[i][sess]:testSessInd[i][sess+1],:] = testX[i][sess]
#             array_trainY[i][trainSessInd[i][sess]:trainSessInd[i][sess+1]] = trainY[i][sess]
#             array_testY[i][testSessInd[i][sess]:testSessInd[i][sess+1]] = testY[i][sess]
            
#     return array_trainX, array_trainY, trainSessInd, array_testX, array_testY, testSessInd

