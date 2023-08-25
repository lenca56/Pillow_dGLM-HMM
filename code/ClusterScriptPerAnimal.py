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
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

ibl_data_path = '../data_IBL'
dfAll = pd.read_csv(ibl_data_path + '/Ibl_processed.csv')
subjectsWitten = np.unique(dfAll[dfAll['lab'] == 'wittenlab']['subject'])

# setting hyperparameters
sigmaList = [0] + [10**x for x in list(np.arange(-3,1,0.5,dtype=float))] + [10**x for x in list(np.arange(1,4,1,dtype=float))]
L2penaltyW = 1
priorDirP = None
maxiter = 300
splitFolds = 4
fitFolds = 1 # actually 4

initParam = 'all' # initializing for best GLM-HMM fit from all animals or subject-specific one

D = 4 # number of features

sessStop = -1 # last session to use in fitting
subject = subjectsWitten[idx] # subject to fit
# fitting for K = 1,2,3,4
for K in [1]:
    x, y, sessInd = get_mouse_design(dfAll, subject, sessStop=sessStop, D=D) # NOT LOOKING AT FULL DATASET

    if (initParam == 'all'):
        glmhmmW = np.load(f'../data_IBL/W_IBL_allAnimals_bestGLMHMM-Iris_D={D}_{K}-state.npy')
        glmhmmP = np.load(f'../data_IBL/P_IBL_allAnimals_bestGLMHMM-Iris_D={D}_{K}-state.npy')
    elif(initParam == 'subject'):
        glmhmmW = np.load(f'../data_IBL/W_IBL_{subject}_bestGLMHMM-Iris_D={D}_{K}-state.npy')
        glmhmmP = np.load(f'../data_IBL/P_IBL_{subject}_bestGLMHMM-Iris_D={D}_{K}-state.npy')

    # fitting
    trainLl, testLl, allP, allW, trainSessInd, testSessInd = fit_eval_CV_multiple_sigmas(x, y, sessInd, K, splitFolds=splitFolds, fitFolds=fitFolds, sigmaList=sigmaList, maxiter=maxiter, glmhmmW=glmhmmW, glmhmmP=glmhmmP, L2penaltyW=L2penaltyW, priorDirP=priorDirP)
        
    # saving
    for fold in range(0, fitFolds):
        np.save(f'../data_IBL/{subject}/trainLl_{subject}_D={D}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', trainLl[fold])
        np.save(f'../data_IBL/{subject}/testLl_{subject}_D={D}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', testLl[fold])
        np.save(f'../data_IBL/{subject}/P_{subject}_D={D}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', allP[fold])
        np.save(f'../data_IBL/{subject}/W_{subject}_D={D}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', allW[fold])
        np.save(f'../data_IBL/{subject}/trainSessInd_{subject}_D={D}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', np.array(trainSessInd[fold]))
        np.save(f'../data_IBL/{subject}/testSessInd_{subject}_D={D}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', np.array(testSessInd[fold]))
