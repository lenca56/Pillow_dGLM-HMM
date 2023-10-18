from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# from oneibl.onelight import ONE # only used for downloading data
# import wget
from utils import *
from plotting_utils import *
from analysis_utils import *
import dglm_hmm1
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import sys
import os

ibl_data_path = '../data_IBL'
dfAll = pd.read_csv(ibl_data_path + '/Ibl_processed.csv')
df = pd.DataFrame(columns=['lab','subject','K']) # in total z=0,143
z = 0
labChosen = ['angelakilab','churchlandlab','wittenlab']
subjectsRemaining = ['IBL-T3','CSHL_001','CSHL_002','CSHL_005','CSHL_007','CSHL_014','CSHL_015','ibl_witten_04','ibl_witten_05','ibl_witten_06','ibl_witten_07','ibl_witten_12','ibl_witten_13','ibl_witten_15','ibl_witten_16','NYU-01','NYU-04','NYU-06']
# for lab in labChosen:
    # subjects = np.unique(dfAll[dfAll['lab'] == lab]['subject']).tolist()
for subject in subjectsRemaining: #subjects: when fitting all animals
    for K in [1,2,3,4]:
        # df.loc[z, 'lab'] = lab
        df.loc[z, 'subject'] = subject
        df.loc[z, 'K'] = K
        z += 1
# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
# lab = df.loc[idx,'lab']
subject = df.loc[idx,'subject']
K = df.loc[idx,'K']

# setting hyperparameters
sigmaList = [0] + [10**x for x in list(np.arange(-3,1,0.5,dtype=float))] + [10**x for x in list(np.arange(1,4,1,dtype=float))]
L2penaltyW = 1
priorDirP = None
maxiter = 300
splitFolds = 5
fit_init_states = False

D = 4 # number of features
sessStop = -1 # last session to use in fitting

# fitting for K = 1,2,3,4
x, y, sessInd, _ = get_mouse_design(dfAll, subject, sessStop=sessStop, D=D) # NOT LOOKING AT FULL DATASET
N = x.shape[0]
presentTrain, presentTest = split_data(N, sessInd, folds=splitFolds, blocks=10, random_state=1)

glmhmmW = np.load(f'../data_IBL/Best_sigma=0_allAnimals_D={D}_{K}-state_W.npy')
glmhmmP = np.load(f'../data_IBL/Best_sigma=0_allAnimals_D={D}_{K}-state_P.npy')

trainLl = np.zeros((splitFolds, len(sigmaList), maxiter))
testLl = np.zeros((splitFolds, len(sigmaList)))
testAccuracy = np.zeros((splitFolds, len(sigmaList)))
allP = np.zeros((splitFolds, len(sigmaList), K, K))
allW = np.zeros((splitFolds, len(sigmaList), N,K,D,2)) 

# fitting
for fold in range(0,splitFolds):    
    allP[fold], _, allW[fold], trainLl[fold], testLl[fold], testAccuracy[fold] = fit_eval_CV_multiple_sigmas(K, x, y, sessInd, presentTrain[fold], presentTest[fold], sigmaList=sigmaList, maxiter=maxiter, glmhmmW=glmhmmW, glmhmmP=glmhmmP, L2penaltyW=L2penaltyW, priorDirP=priorDirP, fit_init_states=fit_init_states)
                                                                
# saving
np.savez(f'../data_IBL/{subject}/{subject}_ALL-PARAM_D={D}_{K}-state_multiple-sigmas_L2penaltyW={L2penaltyW}_untilSession{sessStop}', trainLl=trainLl, allP=allP, allW=allW, testLl=testLl, testAccuracy=testAccuracy)

