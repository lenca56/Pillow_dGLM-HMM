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
df = pd.DataFrame(columns=['subject','fold']) # in total z=0,179 inclusively
splitFolds = 5
labChosen = ['angelakilab','churchlandlab','wittenlab']
z = 0
for lab in labChosen:
    subjects = np.unique(dfAll[dfAll['lab'] == lab]['subject']).tolist()
    for subject in subjects:
        for fold in range(splitFolds):
            df.loc[z, 'subject'] = subject
            df.loc[z, 'fold'] = fold
            z += 1

# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
subject = df.loc[idx,'subject']
fold = df.loc[idx,'fold']

# setting hyperparameters
alphaList = [2*(10**x) for x in list(np.arange(-1,6,0.5,dtype=float))] 
L2penaltyW = 1
maxiter = 200
bestSigma = 1 # verified from fitting multiple sigmas
priorDirP = [100,10] # to read dGLMHMM1 model
fit_init_states = False
K = 3
D = 4 # number of features
sessStop = -1 # last session to use in fitting

# fitting for K = 1,2,3,4
x, y, sessInd, _ = get_mouse_design(dfAll, subject, sessStop=sessStop, D=D) # NOT LOOKING AT FULL DATASET
N = x.shape[0]
presentTrain, presentTest = split_data(N, sessInd, folds=splitFolds, blocks=10, random_state=1)

# parameters for best model in dGLM-HMM1 (only weights varying)
globalP = np.load(f'../data_IBL/{subject}/{subject}_bestP_D={D}_{K}_state_CV_sigma=1_priorDirP={priorDirP}_L2penaltyW={L2penaltyW}_untilSession{sessStop}.npy')
dglmhmmW  = np.load(f'../data_IBL/{subject}/{subject}_bestW_D={D}_{K}_state_CV_sigma=1_priorDirP={priorDirP}_L2penaltyW={L2penaltyW}_untilSession{sessStop}.npy')
    
# fitting
P, _, W, _, testLl, testAccuracy = fit_eval_CV_multiple_alphas(K, x, y, sessInd, presentTrain[fold], presentTest[fold], alphaList=alphaList, maxiter=maxiter, dglmhmmW=dglmhmmW, globalP=globalP, bestSigma=bestSigma, L2penaltyW=L2penaltyW, fit_init_states=fit_init_states)
                                                       
# saving
np.savez(f'../data_IBL/{subject}/{subject}_ALL-PARAM_D={D}_{K}-state_fold-{fold}_multiple-alphas_L2penaltyW={L2penaltyW}_untilSession{sessStop}', P=P, W=W, testLl=testLl, testAccuracy=testAccuracy)
