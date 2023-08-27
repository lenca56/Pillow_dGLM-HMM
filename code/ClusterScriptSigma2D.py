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
# idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
idx = int(sys.argv[1])
subject = df.loc[idx,'subject']
K = df.loc[idx,'K']
fold = df.loc[idx,'fold']


# setting hyperparameters
sigmaList =  [1,10] #[10**x for x in list(np.arange(-3,1,0.5,dtype=float))] + [10**x for x in list(np.arange(-3,1,0.5,dtype=float))] + [10]
L2penaltyW = 1
priorDirP = None
maxiter = 300
splitFolds = 4

initParam = 'all' # initializing for best GLM-HMM fit from all animals or subject-specific one

D = 4 # number of features
stimCol = [1]

sessStop = -1 # last session to use in fitting
subject = subjectsWitten[idx] # subject to fit
x, y, sessInd = get_mouse_design(dfAll, subject, sessStop=sessStop, D=D) # NOT LOOKING AT FULL DATASET
trainX, trainY, trainSessInd, testX, testY, testSessInd = split_data(x, y, sessInd, folds=splitFolds, blocks=10, random_state=1)

if (initParam == 'all'):
    glmhmmW = np.load(f'../data_IBL/W_IBL_allAnimals_bestGLMHMM-Iris_D={D}_{K}-state.npy')
    glmhmmP = np.load(f'../data_IBL/P_IBL_allAnimals_bestGLMHMM-Iris_D={D}_{K}-state.npy')
elif(initParam == 'subject'):
    glmhmmW = np.load(f'../data_IBL/W_IBL_{subject}_bestGLMHMM-Iris_D={D}_{K}-state.npy')
    glmhmmP = np.load(f'../data_IBL/P_IBL_{subject}_bestGLMHMM-Iris_D={D}_{K}-state.npy')

# fitting
trainLl, testLl, allP, allW, trainSessInd, testSessInd = fit_eval_CV_2Dsigmas(trainX[fold], trainY[fold], trainSessInd[fold], testX[fold], testY[fold], testSessInd[fold], K=K, sigmaList=sigmaList, maxiter=maxiter, glmhmmW=glmhmmW, glmhmmP=glmhmmP, L2penaltyW=L2penaltyW, priorDirP=priorDirP, stimCol=stimCol)
     
# saving
np.save(f'../data_IBL/{subject}/trainLl_{subject}_D={D}_{K}_state_fold-{fold}_sigmas2D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', trainLl)
np.save(f'../data_IBL/{subject}/testLl_{subject}_D={D}_{K}_state_fold-{fold}_sigmas2D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', testLl)
np.save(f'../data_IBL/{subject}/P_{subject}_D={D}_{K}_state_fold-{fold}_sigmas2D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', allP)
np.save(f'../data_IBL/{subject}/W_{subject}_D={D}_{K}_state_fold-{fold}_sigmas2D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', allW)
np.save(f'../data_IBL/{subject}/trainSessInd_{subject}_D={D}_{K}_state_fold-{fold}_sigmas2D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', np.array(trainSessInd))
np.save(f'../data_IBL/{subject}/testSessInd_{subject}_D={D}_{K}_state_fold-{fold}_sigmas2D_L2penaltyW={L2penaltyW}_priorDirP={priorDirP}_untilSession{sessStop}_init-{initParam}', np.array(testSessInd))
