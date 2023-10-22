from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# from oneibl.onelight import ONE # only used for downloading data
# import wget
from utils import *
from plotting_utils import *
from analysis_utils import *
import dglm_hmm2
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import sys
import os

ibl_data_path = '../data_IBL'
dfAll = pd.read_csv(ibl_data_path + '/Ibl_processed.csv')
labChosen =  ['angelakilab','churchlandlab','wittenlab']
subjectsAll = []
for lab in labChosen:
    subjects = np.unique(dfAll[dfAll['lab'] == lab]['subject']).tolist()
    subjectsAll = subjectsAll + subjects
# read from cluster array in order to get parallelizations
idx = 0 # int(os.environ["SLURM_ARRAY_TASK_ID"]) # idx=0,35 inclusively
subject = subjectsAll[idx]

# setting hyperparameters
L2penaltyW = 1
maxiter = 200
bestSigma = 1 # verified from fitting multiple sigmas
priorDirP = [100,10]
fit_init_states = False
D = 4 # number of features
sessStop = -1 # last session to use in fitting
bestAlpha = 2 # found by cross-validation
K = 3

# fitting for K = 1,2,3,4
x, y, sessInd, _ = get_mouse_design(dfAll, subject, sessStop=sessStop, D=D) 
N = x.shape[0]
sess = len(sessInd)-1

# parameters for best model in dGLM-HMM1 (only weights varying)
globalP = np.load(f'../data_IBL/{subject}/{subject}_bestP_D={D}_{K}_state_CV_sigma=1_priorDirP={priorDirP}_L2penaltyW={L2penaltyW}_untilSession{sessStop}.npy')
dglmhmmW  = np.load(f'../data_IBL/{subject}/{subject}_bestW_D={D}_{K}_state_CV_sigma=1_priorDirP={priorDirP}_L2penaltyW={L2penaltyW}_untilSession{sessStop}.npy')
    
inits = 21 # first one is constant P
trainLl = np.zeros((inits,maxiter))
testLl = np.zeros((inits))
testAccuracy = np.zeros((inits))
allP = np.zeros((inits,N,K,K))
allW = np.zeros((inits,N,K,D,2))

# creating matrix for initializations of P from globalP with addedd noise
initP = np.zeros((inits,N,K,K))
noiseDir = np.ones((K,K))
alpha = 10
for k in range(0,K):
    noiseDir[k,k] = alpha # diagonal element
# first init is just constant transition matrix P (from dGLMHMM1)
initP[0] = reshapeP_M1_to_M2(globalP, N) 
for init in range(1,inits):
    for s in range(0,sess):
        for k in range(0,K):
            initP[init,sessInd[s]:sessInd[s+1],k,:] = (globalP[k] + np.random.dirichlet(noiseDir[k]))/2
np.save(f'../data_IBL/{subject}/{subject}_initP-noisy-alpha=10_inits={inits}.npy', initP)
initP = np.load(f'../data_IBL/{subject}/{subject}_initP-noisy-alpha=10_inits={inits}.npy')

# if not fitting dGLMHMM1
# allP[0] = initP[0] # globalP repeated across sessions - constant
# allW[0] = np.copy(dglmhmmW)

pi = np.ones((K))/K    
presentAll = np.ones((N))
dGLM_HMM2 = dglm_hmm2.dGLM_HMM2(N,K,D,2)
for init in range(0,inits):
    # fitting
    allP[init], _, allW[init], trainLl[init] = dGLM_HMM2.fit(x, y, presentAll, initP[init], pi, dglmhmmW, sigma=reshapeSigma(bestSigma, K, D), alpha=bestAlpha, globalP=globalP, sessInd=sessInd, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW, fit_init_states=fit_init_states) 

    # evaluate 
    testLl[init], testAccuracy[init] = dGLM_HMM2.evaluate(x, y, sessInd, presentAll, allP[init], pi, allW[init], sortStates=False)
                                 
np.savez(f'../data_IBL/{subject}/{subject}_ALL-PARAM_D={D}_{K}_state_alpha={bestAlpha}_multiple-initsP_L2penaltyW={L2penaltyW}_untilSession{sessStop}', trainLl=trainLl, allP=allP, allW=allW, testLl=testLl, testAccuracy=testAccuracy)
