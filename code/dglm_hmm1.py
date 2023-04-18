# importing packages and modules
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from utils import *
from plotting_utils import *
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
# from autograd import value_and_grad, hessian
import jax

class dGLM_HMM1():
    """
    Class for fitting driftinig GLM-HMM model 1 in which weights are constant within session but vary across sessions
    Code just works for c=2 at the moment
    Weights for class c=1 are always kept to 0 (so then emission probability becomes 1/(1+exp(-wTx)))
    X columns represent [bias, sensory] in this order

    Notation: 
        n: number of data points
        k: number of states (states)
        d: number of features (inputs to design matrix)
        c: number of classes (possible observations)
        X: design matrix (n x d)
        Y: observations (n x c) or (n x 1)
        w: weights mapping x to y (n x k x d x c)
    """

    def __init__(self, n, k, d, c):
            self.n, self.k, self.d, self.c  = n, k, d, c
    
    # Iris' code has reversed columns
    def observation_probability(self, x, w):
        '''
        Calculating observation probabilities for given design matrix x and weight matrix w

        Parameters
        ----------
        x: Ncurrent x D numpy array
            input matrix
        w: Ncurrent x K x D x C numpy array
            weight matrix

        Returns
        -------
        phi: Ncurrent x K x C numpy array
            observation probabilities matrix
        '''
        
        Ncurrent = x.shape[0]

        phi = np.empty((Ncurrent, self.k, self.c)) # probability that it is state 1
        for k in range(0, self.k):
            for c in range(0, self.c):
                phi[:,k,c] = np.exp(-np.sum(w[:,k,:,c]*x,axis=1))
            phi[:,k,:]  = np.divide((phi[:,k,:]).T,np.sum(phi[:,k,:],axis=1)).T     

        return phi

    # def log_observation_probability(self, x, w):
    #     '''
    #     Calculating log observation probabilities for given design matrix x and weight matrix w

    #     Parameters
    #     ----------
    #     x: Ncurrent x D numpy array
    #         input matrix
    #     w: Ncurrent x K x D x C numpy array
    #         weight matrix

    #     Returns
    #     -------
    #     phi: Ncurrent x K x C numpy array
    #         observation probabilities matrix
    #     '''
        
    #     Ncurrent = x.shape[0]

    #     phi = np.empty((Ncurrent, self.k, self.c)) # probability that it is state 1
    #     for k in range(0, self.k):
    #         for c in range(0, self.c):
    #             phi[:,k,c] = np.exp(-np.sum(w[:,k,:,c]*x,axis=1))
    #         phi[:,k,:]  = np.divide((phi[:,k,:]).T,np.sum(phi[:,k,:],axis=1)).T     

    #     return phi
    
    
    def simulate_data(self, trueW, trueP, sessInd, save=False, title='sim', pi0=0.5):
        '''
        function that simulates X and Y data from true weights and true transition matrix

        Parameters
        ----------
        trueW: n x k x d x c numpy array
            true weight matrix. for c=2, trueW[:,:,:,1] = 0 
        trueP: k x k numpy array
            true probability transition matrix
        priorZstart: int
            0.5 probability of starting a session with state 0 (works for C=2)
        sessInd: list of int
            indices of each session start, together with last session end + 1
        save: boolean
            whether to save out simulated data
        pi0: float
            constant between 0 and 1, representing probability that first latent in a session is state 0
            
        Returns
        -------
        x: n x d numpy array
            simulated design matrix
        y: n x 1 numpy array
            simulated observation vector
        z: n x 1 numpy array
            simulated hidden states vector

        '''
        # check that weight and transition matrices are valid options
        if (trueW.shape != (self.n, self.k, self.d, self.c)):
            raise Exception(f'Weights need to have shape ({self.n}, {self.k}, {self.d}, {self.c})')
        
        if (trueP.shape != (self.k, self.k)):
            raise Exception(f'Transition matrix needs to have shape ({self.k}, {self.k})')
        
        x = np.empty((self.n, self.d))
        y = np.zeros((self.n, self.c)).astype(int)
        z = np.empty((self.n,),dtype=int)

        # input data x
        x[:,0] = 1 # bias term
        x[:,1] = stats.uniform.rvs(loc=-16,scale=33,size=self.n).astype(int)
        # standardizing sensory info
        x[:,1] = x[:,1] - x[:,1].mean()
        x[:,1] = x[:,1] / x[:,1].std()

        # TRY ormal distribution for x[:,1]

        if (self.k==1):
            z[:] = 0
        elif (self.k ==2):
            # latent variables z 
            for t in range(0, self.n):
                if (t in sessInd[:-1]): # beginning of session has a new draw for latent
                    z[t] = np.random.binomial(n=1,p=1-pi0)
                else:
                    z[t] = np.random.binomial(n=1, p=trueP[z[t-1],1])
        elif (self.k >=3):
            raise Exception("simulate data does not support k>=3")
        
        # observation probabilities
        phi = self.observation_probability(x, trueW)

        for t in range(0, self.n):
            y[t,int(np.random.binomial(n=1,p=phi[t,z[t],1]))]=1
        
        y = reshapeObs(y) # reshaping from n x c to n x 1

        if (save==True):
            np.save(f'../data/{title}X', x)
            np.save(f'../data/{title}Y', y)
            np.save(f'../data/{title}Z', z)

        return x, y, z

    # already checked with Iris' function that it is correct
    def forward_pass(self, y, P, phi, pi0=None):
        '''
        Calculates alpha scaled as part of the forward-backward algorithm in E-step 
       
        Parameters
        ----------
        y : T x 1 numpy vector 
            vector of observations with values 0,1,..,C-1
        P : k x k numpy array 
            matrix of transition probabilities
        phi : T x k x  c numpy array
            matrix of observation probabilities
        pi0: k x 1 numpy vector
            distribution of first state before it has sesn any data 
        Returns
        -------
        alpha : T x k numpy vector
            matrix of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        ct : T x 1 numpy vector
            vector of the forward marginal likelihoods p(y_t | y_1:t-1)
        ll : float
            marginal log-likelihood of the data p(y)
        '''
        T = y.shape[0]
        
        alpha = np.zeros((T, self.k)) # forward probabilities p(z_t | y_1:t)
        alpha_prior = np.zeros((T, self.k)) # prior probabilities p(z_t | y_1:t-1)
        lt = np.zeros((T, self.k)) # likelihood of data p(y_t|z_t)
        ct = np.zeros(T) # forward marginal likelihoods p(y_t | y_1:t-1)

        # forward pass calculations
        for t in range(0,T):
            lt[t,:] = phi[t,:,y[t]] # likelihood p(y_t | z_t)
            if (t==0): # time point 0
                # prior of z_0 before any data 
                if (pi0==None):
                    alpha_prior[0,:] = np.ones((1,self.k))/self.k # uniform prior
                else:
                    alpha_prior[0,:] = pi0
            else:
                alpha_prior[t,:] = (alpha[t-1,:].T @ P) # conditional p(z_t | y_1:t-1)
            pxz = np.multiply(lt[t],alpha_prior[t,:]) # joint P(y_1:t, z_t)
            ct[t] = np.sum(pxz) # conditional p(y_t | y_1:t-1)
            alpha[t,:] = pxz/ct[t] # conditional p(z_t | y_1:t)
        
        ll = np.sum(np.log(ct)) # marginal log likelihood p(y_1:T) as sum of log conditionals p(y_t | y_1:t-1) 
        
        return alpha, ct, ll
    
    # already checked with Iris' function that it is correct
    def backward_pass(self, y, P, phi, ct, pi0=None):
        '''
        Calculates beta scaled as part of the forward-backward algorithm in E-step 

        Parameters
        ----------
        y : T x 1 numpy vector
            vector of observations with values 0,1,..,C-1
        p : k x k numpy array
            matrix of transition probabilities
        phi : T x k x c numppy array
            matrix of observation probabilities
        ct : T x 1 numpy vector 
            veector of forward marginal likelihoods p(y_t | y_1:t-1), calculated at forward_pass
            
        Returns
        -------
        beta: T x k numpy array 
            matrix of backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)
        '''

        T = y.shape[0]
        
        beta = np.zeros((T, self.k)) # backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)
        lt = np.zeros((T, self.k)) # likelihood of data p(y_t|z_t)

        # last time point
        beta[-1] = 1 # p(z_T=1)

        # backward pass calculations
        for t in np.arange(T-2,-1,-1):
            lt[t+1,:] = phi[t+1,:,y[t+1]] 
            beta[t,:] = P @ (np.multiply(beta[t+1,:],lt[t+1,:]))
            beta[t,:] = beta[t,:] / ct[t+1] # scaling factor
        
        return beta
    
    def posteriorLatents(self, y, p, phi, alpha, beta, ct):
        ''' 
        calculates marginal posterior of latents gamma(z_t) = p(z_t | y_1:T)
        and joint posterior of successive latens zeta(z_t, z_t+1) = p(z_t, z_t+1 | y_1:T)

        Parameters
        ----------
        y : T x 1 numpy vector 
            vector of observations with values 0,1,..,C-1
        p : k x k numpy array
            matrix of transition probabilities
        phi : T x k x c numppy array
            matrix of observation probabilities
        alpha : T x k numpy vector
            marix of the conditional probabilities p(z_t | x_1:t, y_1:t)
        beta: T x k numpy array 
            matrix of backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)
        ct : T x 1 numpy vector
            vector of the forward marginal likelihoods p(y_t | y_1:t-1)
        
        Returns
        -------
        gamma: T x k numpy array
            matrix of marginal posterior of latents p(z_t | y_1:T)
        zeta: T-1 x k x k 
            matrix of joint posterior of successive latens p(z_t, z_t+1 | y_1:T)
        '''
        
        T = ct.shape[0]
        gamma = np.empty((T, self.k)).astype(float) # marginal posterior of latents
        zeta = np.empty((T-1, self.k, self.k)).astype(float) # joint posterior of successive latents

        gamma = np.multiply(alpha, beta) # gamma(z_t) = alpha(z_t) * beta(z_t)

        # zeta(z_t, z_t+1) =  alpha(z_t) * beta(z_t+1) * p (z_t+1 | z_t) * p(y_t+1 | z_t+1) / c_t+1
        for t in range(0,T-1):
            alpha_beta = alpha[t,:].reshape((self.k, 1)) @ beta[t+1,:].reshape((1, self.k))
            zeta[t,:,:] = np.multiply(alpha_beta,p) 
            zeta[t,:,:] = np.multiply(zeta[t,:,:],phi[t+1,:,y[t+1]]) # change t+1 to t in phi to match Iris'
            zeta[t,:,:] = zeta[t,:,:] / ct[t+1]
            
        return gamma, zeta

    def generate_param(self, sessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]):
        ''' 
        Function that generates parameters w and P and is used for initialization of parameters during fitting

        Parameters
        ----------
        sessInd: list of int
            indices of each session start, together with last session end + 1
        transitionDistribution: list of length 2
            first is str of distribution type, second is parameter tuple 
                dirichlet ditribution comes with (alphaDiagonal, alphaOther) the concentration values for either main diagonal or other locations
        weightDistribution: list of length 2
            first is str of distribution type, second is parameter tuple
                uniform distribution comes with (low, high) 
                normal distribution comes with (mean, std)

        Returns
        ----------
        p: k x k numpy array
            probability transition matrix
        w: T x k x d x c numpy array
            weight matrix. for c=2, trueW[:,:,:,1] = 0 
        
        '''
        T = int(sessInd[-1])

        sess = len(sessInd)-1 # number of total sessions

        # initialize weight and transitions
        p = np.empty((self.k, self.k))
        w = np.zeros((T, self.k, self.d, self.c))

        # generating transition matrix 
        if (transitionDistribution[0] == 'dirichlet'):
            (alphaDiag, alphaOther) = transitionDistribution[1]
            for k in range(0, self.k):
                alpha = np.full((self.k), alphaOther)
                alpha[k] = alphaDiag # concentration parameter of Dirichlet for row k
                p[k,:] = np.random.dirichlet(alpha)
        else:
            raise Exception("Transition distribution can only be dirichlet")
        
        # generating weight matrix
        if (weightDistribution[0] == 'uniform'):
            (low, high) = weightDistribution[1]
            for s in range(0,sess):
                rv = np.random.uniform(low, high, (self.k, self.d))
                w[sessInd[s]:sessInd[s+1],:,:,0] = rv
        elif (weightDistribution[0] == 'normal'):
            (mean, std) = weightDistribution[1]
            for s in range(0,sess):
                rv = np.random.normal(mean, std, (self.k, self.d))
                w[sessInd[s]:sessInd[s+1],:,:,0] = rv
        else:
            raise Exception("Weight distribution can only be uniform or normal")

        return p, w 
    
    def weight_loss_function(self, currentW, x, y, gamma, prevW, nextW, sigma, prior=True):
        '''
        weight loss function to optimize the weight in M-step of fitting function is calculated as negative of weighted log likelihood + prior terms 
        coming from drifting wrt neighboring sessions

        L(currentW) = sum_t sum_k gamma(z_t=k) * log p(y_t | z_t=k) + log P(currentW | prevW) + log P(currentW | nextW),
        where gamma matrix are fixed by old parameters but observation probabilities p(y_t | z_t=k) are updated with currentW

        Parameters
        ----------
        currentW: k x d numpy array
            weights of current session for C=0
        x: T x d numpy array
            design matrix
        y : T x 1 numpy vector 
            vector of observations with values 0,1,..,C-1
        gamma: T x k numpy array
            matrix of marginal posterior of latents p(z_t | y_1:T)
        prevW: k x d x c numpy array
            weights of previous session
        nextW: k x d x c numpy array
            weights of next session
        sigma: k x d numpy array
            std parameters of normal distribution for each state and each feature
        
        Returns
        ----------
        -lf: float
            loss function for currentW to be minimized
        '''
        # number of datapoints
        T = x.shape[0]

        # reshaping current session weights from flat to (T, k, d, c)
        # currentW = currentW._value.reshape((self.k, self.d))
        currentW = currentW.reshape((self.k, self.d))
        sessW = np.zeros((T, self.k, self.d, self.c))
        for t in range(0,T):
            sessW[t,:,:,0] = currentW[:,:]

        phi = self.observation_probability(x, sessW) # N x K x C phi matrix calculated with currentW
        logPhi = np.log(phi) # natural log of observation probabilities

        # weighted log likelihood term of loss function
        lf = 0
        for t in range(0, T):
            lf += np.multiply(gamma[t,:],logPhi[t,:,y[t]]).sum()
        #print("Gamma term ",lf)
        
        # prior term for drifting of loss function
        # currentW | prevW ~ Normal(prevW, sigma^2) and currentW | nextW ~ Normal(nextW, sigma^2)
        # for k in range(0, self.k):
        #     if (prevW is not None):
        #         rv = multivariate_normal(mean=prevW[k,:,0], cov=np.diag(np.square(sigma[k,:])), allow_singular=True)
        #         lf += np.log(rv.pdf(currentW[k,:]))   
        #     if (nextW is not None):
        #         rv = multivariate_normal(mean=nextW[k,:,0], cov=np.diag(np.square(sigma[k,:])), allow_singular=True)
        #         lf += np.log(rv.pdf(currentW[k,:]))

        if(prior==True):
            for k in range(0, self.k):
                # inverse of covariance matrix
                invSigma = np.square(1/sigma[k,:])
                det = np.prod(invSigma)
                invCov = np.diag(invSigma)

                if (prevW is not None):
                    # logpdf of multivariate normal (ignoring pi constant)
                    lf +=  -1/2 * np.log(det) - 1/2 * (currentW[k,:] - prevW[k,:,0]).T @ invCov @ (currentW[k,:] - prevW[k,:,0])
                    #print("Prev sess term ", 1/2 * (currentW[k,:] - prevW[k,:,0]).T @ invCov @ (currentW[k,:] - prevW[k,:,0]))
                if (nextW is not None):
                    # logpdf of multivariate normal (ignoring pi constant)
                    lf += -1/2 * np.log(det) - 1/2 * (currentW[k,:] - nextW[k,:,0]).T @ invCov @ (currentW[k,:] - nextW[k,:,0])
                    #print("Next sess term ", 1/2 * (currentW[k,:] - nextW[k,:,0]).T @ invCov @ (currentW[k,:] - nextW[k,:,0]))
                
            # penalty term for size of weights
            #lf -= 1/2 * currentW[k,:].T @ currentW[k,:]

        return -lf
    
    def fit(self, x, y,  initP, initW, sigma, sessInd=None, pi0=None, maxIter=250, tol=1e-3):
        '''
        Fitting function based on EM algorithm. Algorithm: observation probabilities are calculated with old weights for all sessions, then 
        forward and backward passes are done for each session, weights are optimized for one particular session (phi stays the same),
        then after all weights are optimized (in consecutive increasing order), the transition matrix is updated with the old zetas that
        were calculated before weights were optimized

        Parameters
        ----------
        x: T x d numpy array
            design matrix
        y : T x 1 numpy vector 
            vector of observations with values 0,1,..,C-1
        initP :k x k numpy array
            initial matrix of transition probabilities
        initW: n x k x d x c numpy array
            initial weight matrix
        sigma: k x d numpy array
            st dev of normal distr for weights drifting over sessions
        sessInd: list of int
            indices of each session start, together with last session end + 1
        pi0 : k x 1 numpy vector
            initial k x 1 vector of state probabilities for t=1.
        maxiter : int
            The maximum number of iterations of EM to allow. The default is 300.
        tol : float
            The tolerance value for the loglikelihood to allow early stopping of EM. The default is 1e-3.
        
        Returns
        -------
        p: k x k numpy array
            fitted probability transition matrix
        w: T x k x d x c numpy array
            fitteed weight matrix
        ll: float
            marginal log-likelihood of the data p(y)
        '''
        # number of datapoints
        T = x.shape[0]

        if sessInd is None:
            sessInd = [0, T]
            sess = 1 # equivalent to saying the entire data set has one session
        else:
            sess = len(sessInd)-1 # total number of sessions 

        # initialize weights and transition matrix
        w = np.copy(initW)
        p = np.copy(initP)

        # initialize zeta = joint posterior of successive latents 
        zeta = np.zeros((T-1, self.k, self.k)).astype(float) 
        # initialize marginal log likelihood p(y)
        ll = np.zeros((maxIter)).astype(float) 

        #plotting_weights(initW, sessInd, 'initial weights')

        for iter in range(maxIter):
            print(iter)

            # first iteration is without the drifting prior to speed up solution finding
            if(iter==0):
                prior=False
            else:
                prior=True
            
            # calculate observation probabilities given theta_old
            phi = self.observation_probability(x, w)

            # EM step for each session independently 
            for s in range(0,sess):
                
                # E step - forward and backward passes given theta_old (= previous w and p)
                alphaSess, ctSess, llSess = self.forward_pass(y[sessInd[s]:sessInd[s+1]], p, phi[sessInd[s]:sessInd[s+1],:,:], pi0=pi0)
                betaSess = self.backward_pass(y[sessInd[s]:sessInd[s+1]], p, phi[sessInd[s]:sessInd[s+1],:,:], ctSess, pi0=pi0)
                gammaSess, zetaSess = self.posteriorLatents(y[sessInd[s]:sessInd[s+1]], p, phi[sessInd[s]:sessInd[s+1],:,:], alphaSess, betaSess, ctSess)
                
                # merging zeta for all sessions 
                zeta[sessInd[s]:sessInd[s+1]-1,:,:] = zetaSess[:,:,:] 
                ll[iter] += llSess
                
                # M step for weights - weights are updated for each session individually (as neighboring session weights have to be fixed)
                prevW = w[sessInd[s-1]] if s!=0 else None # k x d x c matrix of previous session weights
                nextW = w[sessInd[s+1]] if s!=sess-1 else None # k x d x c matrix of next session weights
                w_flat = np.ndarray.flatten(w[sessInd[s],:,:,0]) # flatten weights for optimization 
                optimized = minimize(self.weight_loss_function, w_flat, args=(x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], gammaSess, prevW, nextW, sigma, prior))
                optimizedW = np.reshape(optimized.x,(self.k, self.d)) # reshape optimized weights
                w[sessInd[s]:sessInd[s+1],:,:,0] = optimizedW # updating weight w for current session

                # optimizedW = np.zeros((self.k,self.d,self.c))
                # # prevW = w[sessInd[s-1]] if s!=0 else None # k x d x c matrix of previous session weights
                # # nextW = w[sessInd[s+1]] if s!=sess-1 else None # k x d x c matrix of next session weights
                # # optimized = minimize(self.weight_loss_function, w_flat, args=(x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], gammaSess, prevW, nextW, sigma))
                # for k in range(0, self.k):
                #     prevW = w[sessInd[s-1],k,:,:] if s!=0 else None # k x d x c matrix of previous session weights
                #     nextW = w[sessInd[s+1],k,:,:] if s!=sess-1 else None # k x d x c matrix of next session weights
                #     w_flat = np.ndarray.flatten(w[sessInd[s],k,:,0]) # flatten weights for optimization 
                #     opt_log = lambda w: self.weight_loss_function_one_state(w, x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], gammaSess[:,k], prevW, nextW, sigma[k]) # calculate log likelihood 
                #     optimized = minimize(value_and_grad(opt_log), w_flat) # , jac = "True", method = "L-BFGS-B")
                #     optimizedW[k,:,0] = np.reshape(optimized.x,(1, self.d)) # reshape optimized weights
                # w[sessInd[s]:sessInd[s+1],:,:,0] = optimizedW # updating weight w for current session
            
            #plotting_weights(w, sessInd, f'iter {iter} optimized')
            # print(w[sessInd[:-1]])

            # M-step for transition matrix p - for all sessions together
            for i in range(0, self.k):
                for j in range(0, self.k):
                    p[i,j] = zeta[:,i,j].sum()/zeta[:,i,:].sum() # closed form update
        
            # check if stopping early 
            if (iter >= 10 and ll[iter] - ll[iter-1] < tol):
                break

        return p, w, ll
    
    def split_data(self, x, y, sessInd, folds=10, random_state=1):
        ''' 
        splitting data function for cross-validation
        currently does not balance trials for each session

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
        # initializing test and train size based on number of folds
        train_size = int(self.n - self.n/folds)
        test_size = int(self.n/folds)

        # initializing input and output arrays for each folds
        trainY = np.zeros((folds, train_size)).astype(int)
        testY = np.zeros((folds, test_size)).astype(int)
        trainX = np.zeros((folds, train_size, self.d))
        testX = np.zeros((folds, test_size, self.d))

        # splitting data for each fold
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        for i, (train_index, test_index) in enumerate(kf.split(y)):
            trainY[i,:], testY[i,:] = y[train_index], y[test_index]
            trainX[i,:,:], testX[i,:,:] = x[train_index], x[test_index]
        
        # initializing session indices for each fold
        trainSessInd = [[0] for i in range(0, folds)]
        testSessInd = [[0] for i in range(0, folds)]

        # getting sesssion start indices for each fold
        for i, (train_index, test_index) in enumerate(kf.split(y)):
            for sess in range(1,len(sessInd)-1):
                testSessInd[i].append(np.argmin(test_index < sessInd[sess]))
                trainSessInd[i].append(np.argmin(train_index < sessInd[sess]))
            testSessInd[i].append(test_index.shape[0])
            trainSessInd[i].append(train_index.shape[0])
        
        return trainX, trainY, trainSessInd, testX, testY, testSessInd

  

        
