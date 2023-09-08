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

inits = 20
D = 4

df = pd.DataFrame(columns=['init','K', 'sign']) # in total z=0,159
z = 0
for init in range(0,inits):
    for K in [1,2,3,4]:
        for sign in [-1, +1]:
            df.loc[z, 'init'] = init
            df.loc[z, 'K'] = K
            df.loc[z, 'sign'] = sign
            z += 1

x = np.load(f'../data_IBL/X_allAnimals_D={D}.npy')
y = np.load(f'../data_IBL/Y_allAnimals_D={D}.npy')
sessInd = np.load(f'../data_IBL/sessInd_allAnimals_D={D}.npy')

# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
init = df.loc[idx,'init']
K = df.loc[idx,'K']
sign = df.loc[idx, 'sign']
x[:,1] = sign * x[:,1] # if sign=1 then x = Right - Left, if sign=-1 then x = Left - Right

N = x.shape[0]
C=2
maxiter = 250
oneSessInd = [0,N]
present = np.ones((N)).astype(int) # using all data
dGLM_HMM = dglm_hmm1.dGLM_HMM1(N,K,D,C)
initP0, initpi0, initW0 = dGLM_HMM.generate_param(sessInd=oneSessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]) 
P, pi, W, _ = dGLM_HMM.fit(x, y, present, initP0, initpi0, initW0, sigma=reshapeSigma(0, K, D), sessInd=sessInd, maxIter=maxiter, tol=1e-4, L2penaltyW=1, priorDirP=None, fit_init_states=False) # sigma does not matter here
ll = dGLM_HMM.evaluate(x, y, present, P, pi, W, sessInd, sortWeights=True) # sorting states too

np.save(f'../data_IBL/all/Ll_allAnimals_D={D}_{K}_state_init-{init}_sign={sign}_sigma=0', ll)
np.save(f'../data_IBL/all/P_allAnimals_D={D}_{K}_state_init-{init}_sign={sign}_sigma=0', P)
np.save(f'../data_IBL/all/W_allAnimals_D={D}_{K}_state_init-{init}_sign={sign}_sigma=0', W)
