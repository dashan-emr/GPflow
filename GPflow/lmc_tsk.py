import GPflow
import numpy as np
import tensorflow as tf
from GPflow.model import AutoFlow
from GPflow.mean_functions import Zero
from GPflow.param import *
import transforms
import conditionals
import kullback_leiblers
from tf_hacks import eye
import pdb

class LMC_TSK(GPflow.model.Model):
    def __init__(self, X, Y,num_tasks,rank,num_latent_list,kern_list, likelihood_list,
                  Z = None, TSK = False, TSKern_list=None, mean_function=Zero(),q_diag_list=None,
                  whiten_list = None,
                  needs_recompile = True,num_inducing_points = 10,
                  W = None, Kappa = None):
        """
        ***** LMC + Task-Specific Kernel (TSK) *****
        X is a data matrix, size N x (D+1) the last column indicates job number
        Y is a data matrix, size N x R
        num_models is number of latent functions in LCM structure
        kern_list is the list of kernels for latent functions
        kern, likelihood, mean_function are appropriate GPflow objects
        Z is a matrix of pseudo inputs, size M x D
        num_latent is the number of latent process to use, default to Y.shape[1]
        q_diag is a boolean. If True, the covariance is approximated by a diagonal matrix.
        whiten is a boolean. It True, we use the whitened represenation of the inducing points.
        """

        assert rank == len(num_latent_list), \
           "length of latent function list should match the rank of the lmc model"
        assert rank == len(kern_list), \
           "length of kernel list should match the rank of the lmc model"
        self.tsk = TSK
        
        if self.tsk:
            self.num_latent = np.sum(num_latent_list,dtype = np.int64) + num_tasks
        else:
            self.num_latent = np.sum(num_latent_list,dtype = np.int64)
            
        self.num_latent_list = num_latent_list
        
        GPflow.model.Model.__init__(self,name = 'LMC_TSK')
        self.mean_function_list = [mean_function]* self.num_latent
        if whiten_list is None: whiten_list = [False] * self.num_latent
        if q_diag_list is None: q_diag_list = [False] * self.num_latent
        self.q_diag_list,self.whiten_list = q_diag_list, whiten_list
        self.X, self.Y = list(),list()
        #self.X = tf.placeholder(tf.float32,shape = [None,1])
        #self.Y = tf.placeholder(tf.float32,shape = [None,1])
        """
        q_sqrt is a 3D tensor, each matrix within is a lower triangular square-root
        matrix of the covariance of q. num_tasks is the number of tasks , num_latent is the
        number of latent functions for each task. Assume here the number of latent function
        of each task is the same.
        """
        self.q_sqrt_list,self.q_mu_list,self.Z,self.kern_list, self.tskern_list,self.likelihood= \
            ParamList([]),ParamList([]),ParamList([]),ParamList([]),ParamList([]),ParamList([])

        self.num_inducing_list,self.num_tasks,self.rank = list(),num_tasks,rank
        """
        Here our default assumption is dimemsion is 1
        """
        self.dim = 1

        locMax ,locMin,num_points = 0,0,100

        """
        The following loop tracks the min and max time over all tasks and
        set the inducing points on grid
        """
        for j in range(self.num_tasks):
            self.X.append(X[X[:,X.shape[1]-1] == j,:X.shape[1]-1])
            self.Y.append(Y[X[:,X.shape[1]-1] == j,:])
            loc_max,loc_min,points = np.max(self.X[j]),np.min(self.X[j]),self.X[j].shape[0]
            if loc_max > locMax: locMax = loc_max
            if loc_min < locMin: locMin = loc_min
            if points < num_points:  num_points = points
            
        for i in range(self.rank):
            self.kern_list.append(kern_list[i])
         
        for k in range(self.num_tasks):
            self.likelihood.append(likelihood_list[k])              
        """
        Assign kernel to each task specific function
        """
        if self.tsk:
            for i in range(self.num_tasks):
                self.tskern_list.append(TSKern_list[i])
                
        for i in range(self.num_latent):
            if Z is None: self.Z.append(Param(np.linspace(locMin,locMax,num_inducing_points)[:,np.newaxis])) #10 is the default number of inducing points
            else: self.Z.append(Param(Z[i]))
            self.num_inducing_list.append(self.Z[i].shape[0])   #initializing inducing points
            if self.q_diag_list[i]:
                self.q_sqrt_list.append(Param(np.ones((self.num_inducing_list[i], self.dim)),transforms.positive))
            else:
                self.q_sqrt_list.append(Param(np.array([np.eye(self.num_inducing_list[i]) for _ in range(Y.shape[1])]).swapaxes(0,2)))
            self.q_mu_list.append(Param(np.zeros((self.num_inducing_list[i], self.dim)))) # dimension of Y is 1
        if W is None:
            self.W = Param(np.random.randn(self.num_tasks,self.num_latent-self.num_tasks))  #dim n * m
        else:
            self.W = W
        if Kappa is None:
            self.Kappa = Param(np.random.randn(self.num_tasks,1))  #dim n * m]
        else:
            self.Kappa = Kappa

    def build_prior_KL(self):
        """
        We return the KL for all latent funtions
        """
        KL = 0
        for i in np.arange(self.rank):  # i is the group id.
            for j in np.arange(self.num_latent_list[i]):
                lat_id = np.sum(self.num_latent_list[:i],dtype = np.int64) + j #id of latent function
                if self.whiten_list[lat_id]:
                    if self.q_diag_list[lat_id]:
                        KL +=  kullback_leiblers.gauss_kl_white_diag(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], self.dim)#rotates the coordinate system to make it independent
                    else:
                        KL += kullback_leiblers.gauss_kl_white(self.q_mu_list[lat_id],self.q_sqrt_list[lat_id], self.dim)
                else:
                    K = self.kern_list[i].K(self.Z[lat_id]) + eye(self.num_inducing_list[lat_id]) * 1e-6 ## compute with the ith kernel
                    if self.q_diag_list[lat_id]:
                        KL += kullback_leiblers.gauss_kl_diag(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], K, self.dim)
                    else:
                        KL += kullback_leiblers.gauss_kl(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], K, self.dim)
        if self.tsk:
            for task_id in np.arange(self.num_tasks):
                lat_id = np.sum(self.num_latent_list,dtype = np.int64) + task_id#id of latent function
                if self.whiten_list[lat_id]:
                    if self.q_diag_list[lat_id]:
                        KL +=  kullback_leiblers.gauss_kl_white_diag(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], self.dim)
                            #rotates the coordinate system to make it independent
                    else:
                        KL += kullback_leiblers.gauss_kl_white(self.q_mu_list[lat_id],self.q_sqrt_list[lat_id], self.dim)
                else:
                    K = self.tskern_list[task_id].K(self.Z[lat_id]) + eye(self.num_inducing_list[lat_id]) * 1e-6 ## compute with the ith kernel
                    if self.q_diag_list[lat_id]:
                        KL += kullback_leiblers.gauss_kl_diag(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id],K, self.dim)
                    else:
                        KL += kullback_leiblers.gauss_kl(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id],K, self.dim)
        return KL

    def build_likelihood(self):
        """
        Loglikelihood is the likelihood sum over all tasks
        the outer loop go through all the tasks and the inner loop goes through
        the latent function inside each task and calculate the loglike of this task
        as the weighted sum of loglike of latent function.
        """
        #Get prior KL.
        KL,loglike  = self.build_prior_KL(),0
        #Get conditionals
        for i in np.arange(self.num_tasks):
            ve,Fmean,Fvar = 0,0,0
            for j in np.arange(self.rank):
                for k in np.arange(self.num_latent_list[j]):
                    lat_id = np.sum(self.num_latent_list[:j],dtype = np.int64) + k
                    if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                        fmean, fvar = conditionals.gaussian_gp_predict_whitened(self.X[i], self.Z[lat_id],
                                        self.kern_list[j], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                    else:
                        fmean, fvar = conditionals.gaussian_gp_predict(self.X[i], self.Z[lat_id],
                                        self.kern_list[j], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                    Fmean += (fmean + self.mean_function_list[lat_id](self.X[i]))*self.W[i,lat_id]
                    Fvar += fvar * tf.square(self.W[i,lat_id])       
            if self.tsk:
                lat_id = np.sum(self.num_latent_list,dtype = np.int64) + i
                if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                    fmean, fvar = conditionals.gaussian_gp_predict_whitened(self.X[i], self.Z[lat_id],
                                    self.tskern_list[i], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                else:
                    fmean, fvar = conditionals.gaussian_gp_predict(self.X[i], self.Z[lat_id],
                                    self.tskern_list[i], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                Fmean += (fmean + self.mean_function_list[lat_id](self.X[i]))*self.Kappa[i,0]
                Fvar += fvar * tf.square(self.Kappa[i,0])
                
            ve = self.likelihood[i].variational_expectations(Fmean, Fvar, self.Y[i])
            loglike += tf.reduce_sum(ve)             
        loglike -= KL
        return loglike

    def build_predict(self,Xnew,task_ind):
            """
            We need to assume the task_ind starts from 0
            """
            Fmean,Fvar = 0,0
            for i in np.arange(self.rank):
                for j in np.arange(self.num_latent_list[i]):
                    lat_id = np.sum(self.num_latent_list[:i],dtype = np.int64) + j
                    if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                        fmean, fvar = conditionals.gaussian_gp_predict_whitened(Xnew, self.Z[lat_id],
                                        self.kern_list[i], self.q_mu_list[lat_id],
                                         self.q_sqrt_list[lat_id], 1)
                    else:
                        fmean, fvar = conditionals.gaussian_gp_predict(Xnew, self.Z[lat_id],
                                        self.kern_list[i], self.q_mu_list[lat_id],
                                         self.q_sqrt_list[lat_id],1)
                    W_ij = tf.gather(self.W,task_ind)[lat_id]
                    Fmean += (fmean + self.mean_function_list[lat_id](Xnew))*W_ij
                    Fvar += fvar * tf.square(W_ij)
            if self.tsk:
                for i in np.arange(self.num_tasks):
                    lat_id = np.sum(self.num_latent_list,dtype = np.int64) + i
                    if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                        fmean, fvar = conditionals.gaussian_gp_predict_whitened(Xnew, self.Z[lat_id],
                                        self.tskern_list[i], self.q_mu_list[lat_id],
                                         self.q_sqrt_list[lat_id], 1)
                    else:
                        fmean, fvar = conditionals.gaussian_gp_predict(Xnew, self.Z[lat_id],
                                        self.tskern_list[i], self.q_mu_list[lat_id],
                                         self.q_sqrt_list[lat_id], 1)
                    switch = tf.cast(tf.equal(tf.to_int64(i), task_ind),tf.float64)
                    W_ij = tf.gather(self.Kappa,i)[0]*switch
                    Fmean += (fmean + self.mean_function_list[lat_id](Xnew))*W_ij
                    Fvar += fvar * tf.square(W_ij)
            return Fmean, Fvar

    @AutoFlow(tf.placeholder(tf.float64, [None, None]), tf.placeholder(tf.int64, []),tf.placeholder(tf.int64,[]))
    def predict_latent(self, Xnew, group_id, ingroup_id):
        """
        Compute the posterior for one of the latent functions.
        """
        Fmean, Fvar = 0,0
        for i in np.arange(self.rank):
            switch1 = tf.cast(tf.equal(i, group_id),tf.float64)
            for j in np.arange(self.num_latent_list[i]):
                lat_id = np.sum(self.num_latent_list[:i],dtype = np.int64) + j
                if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                    fmean, fvar = conditionals.gaussian_gp_predict_whitened(Xnew, self.Z[lat_id],
                                    self.kern_list[i], self.q_mu_list[lat_id],
                                     self.q_sqrt_list[lat_id], 1)
                else:
                    fmean, fvar = conditionals.gaussian_gp_predict(Xnew, self.Z[lat_id],
                                    self.kern_list[i], self.q_mu_list[lat_id],
                                     self.q_sqrt_list[lat_id], 1)
                    switch2 = tf.cast(tf.equal(j, ingroup_id),tf.float64)
                    Fmean += (fmean + self.mean_function_list[lat_id](Xnew))*switch1*switch2
                    Fvar += fvar *switch1*switch2
        return Fmean, Fvar

    @AutoFlow(tf.placeholder(tf.float64),tf.placeholder(tf.int64, []))
    def predict_y(self, Xnew, task_ind):
            """
            Compute the mean and variance of held-out data at the points Xnew
            """
            pred_f_mean, pred_f_var = self.build_predict(Xnew,task_ind)
            Ymean, Yvar = 0,0
            for i in range(self.num_tasks):
                switch_t = tf.cast(tf.equal(task_ind,tf.to_int64(i)),tf.float64)
                ymean,yvar= self.likelihood[i].predict_mean_and_var(pred_f_mean, pred_f_var)
                Ymean += ymean*switch_t
                Yvar  += yvar*switch_t
            return Ymean, Yvar

    @AutoFlow(tf.placeholder(tf.float64),tf.placeholder(tf.int64, []))
    def predict_f(self,Xnew,task_ind):
        """
        We need to assume the task_ind starts from 0
        """
        return self.build_predict(Xnew,task_ind)
