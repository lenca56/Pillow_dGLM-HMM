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
subjectsWitten = np.unique(dfAll[dfAll['lab'] == 'wittenlab']['subject'])
splitFolds=5

df = pd.DataFrame(columns=['subject','K','fold']) # in total z=0,219
z = 0
for subject in subjectsWitten:
    for K in [1,2,3,4]:
        for fold in range(splitFolds):
            df.loc[z, 'subject'] = subject
            df.loc[z, 'K'] = K
            df.loc[z, 'fold'] = fold
            z += 1

# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
subject = df.loc[idx,'subject']
K = df.loc[idx,'K']
fold = df.loc[idx,'fold']

# setting hyperparameters
alphaList = [2*(10**x) for x in list(np.arange(-1,6,0.5,dtype=float))] 
sigmaList = [0] + [10**x for x in list(np.arange(-3,1,0.5,dtype=float))] + [10**x for x in list(np.arange(1,4,1,dtype=float))]
L2penaltyW = 1
maxiter = 200
bestSigma = 1 # verified from fitting multiple sigmas
priorDirP = None
fit_init_states = False

D = 4 # number of features
sessStop = -1 # last session to use in fitting

# fitting for K = 1,2,3,4
x, y, sessInd = get_mouse_design(dfAll, subject, sessStop=sessStop, D=D) # NOT LOOKING AT FULL DATASET
N = x.shape[0]
presentTrain, presentTest = split_data(N, sessInd, folds=splitFolds, blocks=10, random_state=1)

# parameters for best model in dGLM-HMM1 (only weights varying)
dglmhmmW = np.load(f'../data_IBL/{subject}_bestW_D={D}_{K}_state_CV_sigma=1_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}.npy')
globalP = np.load(f'../data_IBL/{subject}_bestP_D={D}_{K}_state_CV_sigma=1_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}.npy')  

# fitting
P, pi, W, trainLl, testLl, testAccuracy = fit_eval_CV_multiple_alphas(K, x, y, sessInd, presentTrain[fold], presentTest[fold], alphaList=alphaList, maxiter=maxiter, dglmhmmW=dglmhmmW, globalP=globalP, bestSigma=bestSigma, L2penaltyW=1, fit_init_states=fit_init_states)
                                                 
# saving
np.save(f'../data_IBL/{subject}/{subject}_trainLl_D={D}_{K}_state_fold-{fold}_alphas_L2penaltyW={L2penaltyW}_untilSession{sessStop}', trainLl)
np.save(f'../data_IBL/{subject}/{subject}_testLl_D={D}_{K}_state_fold-{fold}_alphas_L2penaltyW={L2penaltyW}_untilSession{sessStop}', testLl)
np.save(f'../data_IBL/{subject}/{subject}_testAccuracy_D={D}_{K}_state_fold-{fold}_alphas_L2penaltyW={L2penaltyW}_untilSession{sessStop}', testAccuracy)
np.save(f'../data_IBL/{subject}/{subject}_P_D={D}_{K}_state_fold-{fold}_alphas_L2penaltyW={L2penaltyW}_untilSession{sessStop}', P)
if (fit_init_states==True):
    np.save(f'../data_IBL/{subject}/{subject}_pi_{subject}_D={D}_{K}_state_fold-{fold}_alphas_L2penaltyW={L2penaltyW}_untilSession{sessStop}', pi)
np.save(f'../data_IBL/{subject}/{subject}_W_D={D}_{K}_state_fold-{fold}_alphas_L2penaltyW={L2penaltyW}_untilSession{sessStop}', W)
