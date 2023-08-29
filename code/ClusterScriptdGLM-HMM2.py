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

df = pd.DataFrame(columns=['subject','K','fold']) # in total z=0,131
z = 0
for subject in subjectsWitten:
    for K in [1,2,3]:
        for fold in [0,1,2,3]:
            df.loc[z, 'subject'] = subject
            df.loc[z, 'K'] = K
            df.loc[z, 'fold'] = fold
            z += 1

# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
# idx = int(sys.argv[1])
subject = df.loc[idx,'subject']
K = df.loc[idx,'K']
fold = df.loc[idx,'fold']

# setting hyperparameters
alphaList = [4**x for x in list(np.arange(-1,9,1, dtype=float))] 
L2penaltyW = 1
priorDirP = None
maxiter = 300
splitFolds = 4
fitFolds = 4 

D = 4 # number of features
sessStop = -1 # last session to use in fitting

# fitting for K = 1,2,3,4
x, y, sessInd = get_mouse_design(dfAll, subject, sessStop=sessStop, D=D) # NOT LOOKING AT FULL DATASET

# parameters for best model in dGLM-HMM1 (only weights varying)
dglmhmm1W = np.load(f'')
dglmhmm1P = np.load(f'')

# fitting
# trainLl, testLl, allP, allW, trainSessInd, testSessInd = fit_eval_CV_multiple_sigmas(x, y, sessInd, K, splitFolds=splitFolds, fitFolds=fitFolds, sigmaList=sigmaList, maxiter=maxiter, glmhmmW=glmhmmW, glmhmmP=glmhmmP, L2penaltyW=L2penaltyW, priorDirP=priorDirP)
        
# # saving
# for fold in range(0, fitFolds):
#     np.save(f'../data_IBL/{subject}/trainLl_{subject}_D={D}_{K}_state_fold-{fold}_sigmas1D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', trainLl[fold])
#     np.save(f'../data_IBL/{subject}/testLl_{subject}_D={D}_{K}_state_fold-{fold}_sigmas1D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', testLl[fold])
#     np.save(f'../data_IBL/{subject}/P_{subject}_D={D}_{K}_state_fold-{fold}_sigmas1D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', allP[fold])
#     np.save(f'../data_IBL/{subject}/W_{subject}_D={D}_{K}_state_fold-{fold}_sigmas1D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', allW[fold])
#     np.save(f'../data_IBL/{subject}/trainSessInd_{subject}_D={D}_{K}_state_fold-{fold}_sigmas1D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', np.array(trainSessInd[fold]))
#     np.save(f'../data_IBL/{subject}/testSessInd_{subject}_D={D}_{K}_state_fold-{fold}_sigmas1D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', np.array(testSessInd[fold]))
