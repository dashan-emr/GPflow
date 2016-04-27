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
import sys
import pdb

class ManyLMC(GPflow.model.Model):
    def __init__(self, X, Y,id_list,num_tasks,rank,num_latent_list,kern_list,likelihood_list,
                  Z = None, TSK = False, TSKern_list=None, mean_function=Zero(),q_diag_list=None,
                  whiten_list = None,needs_recompile = True,num_inducing_points = 10,fixed_inducing_points=True):
        """
        ***** LMC + Task-Specific Kernel (TSK) *****
        X is a data matrix, size N x (D+2), the last but two column indicates job number
        the last column indicates patient id.
        Y is a data matrix, size N x 1.
        id_list is the list of patient id.
        rank is the number (rank or group) of kernel functions. 
        num_latent_list is the list of number of functions associated with each kernel, so the length of num_latent_list should be same as the rank. 
        kern_list is a list of kernel objectsfor latent functions, the length of the list shoud be equal to rank.
        likelihood_list is a list of likelihood objects and the length should be of num_tasks * num_paitents
        TSK is boolean.
        TSKern_list is the list of kernels for task specific latent function. the length of the list should be of num_tasks * num_patients.
        Z is a matrix of pseudo inputs, size M x D
        num_latent is the number of latent process to use, default to Y.shape[1]
        q_diag is a boolean. If True, the covariance is approximated by a diagonal matrix.
        whiten is a boolean. It True, we use the whitened represenation of the inducing points.
        num_inducing_points  is the number of grid inducing points .
        """

        assert rank == len(num_latent_list), \
           "length of latent function list should match the rank of the lmc model"
        assert rank == len(kern_list), \
           "length of kernel list should match the rank of the lmc model"

        self.tsk = TSK
        self.num_patients = len(id_list)
        if self.tsk:
            self.num_latent = np.sum(num_latent_list,dtype = np.int64)*self.num_patients + num_tasks * self.num_patients
        else:
            self.num_latent = np.sum(num_latent_list,dtype = np.int64)*self.num_patients
            
        self.num_latent_list = num_latent_list
        GPflow.model.Model.__init__(self,name = 'ManyLMC')
        self.mean_function_list = [mean_function]* self.num_latent
        if whiten_list is None: whiten_list = [False] * self.num_latent
        if q_diag_list is None: q_diag_list = [False] * self.num_latent
        print "chk",num_inducing_points
        self.q_diag_list,self.whiten_list,self.id_list = q_diag_list, whiten_list,id_list
        self.X, self.Y = list(),list()
        
        """
        VERY IMPORTANT: 
            
        Here the q_sqrt_list has following structure: assume we have 3 patients and 2 tasks for each patient, 
        2 kernel groups, each kernel has 2 latent functions.The q_sqrt_list will have following structure:
        since we have two kernels and two latent fun for each kernel and thus we have (2+2)*3 = 12 latent functions AND
        3*2 = 6 (task_specific function,2 tasks for each patient and we have 3 patients here), 
        then we have 18 latent function in total, the length of q_sqrt_list will be 18.
        
        We store latent function in the following order: 
        The list will store first 12 latent funs followed by 6 tsk specific funs.
        the ith latent fun of kth kerenl group of patient p(assuming index start with 0) will be:
        num_latent = np.sum(num_latent_list) (this is the total number of latent function(not count tsk specific latent) of each pateint) 
        for pth patient and the latent fun in kth kernel at position i:
        q_sqrt = q_sqrt_list[p*num_latent + np.sum(num_latent_list[:k]) + i]
        
        For the tsk_specific lat fun for patient p at task j,suppose we have three patients in total
        q_sqrt = q_sqrt[3*num_latent + p*num_tasks + j]
        """   
        
        self.q_sqrt_list,self.q_mu_list,self.Z,self.kern_list, self.tskern_list,self.likelihood= \
            ParamList([]),ParamList([]),ParamList([]),ParamList([]),ParamList([]),ParamList([])
 
        """
        Here our default assumption is data dimemsion is 1. (time sereis data)
        """
        
        self.num_inducing_list,self.num_tasks,self.rank ,self.dim= list(),num_tasks,rank,1
        
        """
        The following loop tracks the min and max time over all tasks and
        set the inducing points on grid. We also asssume our input data is of N*3 dimension and the third column
        indicates the patient id and the last but two column indicates the task id.  
        """


        locMax ,locMin = -1e100,1e100 #HS: init was to zero; it wouldn't find min (max) if all x's are postive (negative)
        for p,ptid in enumerate(self.id_list):
            Xs = X[X[:,X.shape[1]-1] == float(ptid),:X.shape[1]-1]  # extract data X at ptid and discard the id column
            Ys = Y[X[:,X.shape[1]-1] == float(ptid),:Y.shape[1]]
            for d in range(self.num_tasks):
                self.X.append(Xs[Xs[:,Xs.shape[1]-1] == d,:Xs.shape[1]-1]) # discard the task column
                self.Y.append(Ys[Xs[:,Xs.shape[1]-1] == d,:])
                loc_max,loc_min,points = np.max(self.X[d]),np.min(self.X[d]),self.X[d].shape[0]
                if loc_max > locMax: locMax = loc_max
                if loc_min < locMin: locMin = loc_min
                
        #update kernel for kernel_list
        for q in range(self.rank):
            self.kern_list.append(kern_list[q])
         
        ## update likelihood       
        for r in range(num_tasks):
            self.likelihood.append(likelihood_list[r])              
        """
        Assign kernel to each task specific function
        """
        if self.tsk:
            for d in range(self.num_tasks*self.num_patients):
                self.tskern_list.append(TSKern_list[d])
        
        """
        initialize q_mu, q_sqrt and Z.
        """
        #initializing inducing points
        self.Z.append(Param(np.linspace(locMin,locMax,num_inducing_points)[:,np.newaxis])) #10 is the default number of inducing points
        # precomputing the distance
        self.dZ = list()
        self.dZ.append(self.euclid_dist(np.linspace(locMin,locMax,num_inducing_points)[:,np.newaxis],
                                                 np.linspace(locMin,locMax,num_inducing_points)[:,np.newaxis]))
        self.dZX = list()
        for p, ptid in enumerate(self.id_list):
            for d in np.arange(self.num_tasks):
                data_id = self.num_tasks * p + d
                self.dZX.append(self.euclid_dist(np.linspace(locMin,locMax,num_inducing_points)[:,np.newaxis],self.X[data_id]))

        for q in range(self.num_latent):
            self.num_inducing_list.append(self.Z[0].shape[0])
            if self.q_diag_list[q]:
                self.q_sqrt_list.append(Param(np.ones((self.num_inducing_list[q], self.dim)),transforms.positive))
            else:
                self.q_sqrt_list.append(Param(np.array([np.eye(self.num_inducing_list[q]) for _ in range(Y.shape[1])]).swapaxes(0,2)))
            #self.q_mu_list.append(Param(np.zeros((self.num_inducing_list[q], self.dim)))) # dimension of Y is 1
            self.q_mu_list.append(Param(0.1*np.random.randn(self.num_inducing_list[q], self.dim))) # dimension of Y is 1
        
        self.num_latent_shared = np.sum(self.num_latent_list,dtype = np.int64)
        self.W = Param(np.random.randn(self.num_tasks,self.num_latent_shared))  #dim n * m, and we assume W is shared accorss the patients.
        self.Kappa = Param(np.random.randn(self.num_tasks*self.num_patients,1))  #dim (n *m)* 1 n is number of tasks and m is number of patients
        self.laggings = Param(np.random.randn(self.num_tasks-1,1))
        if fixed_inducing_points:
            self.Z.fixed = True

    def square_dist(self, X, X2):
        X = X
        Xs = np.sum(np.square(X), 1)
        if X2 is None:
            return -2*np.matmul(X, np.transpose(X)) + np.reshape(Xs, (-1,1)) + np.reshape(Xs, (1,-1))
        else:
            X2 = X2
            X2s = np.sum(np.square(X2), 1)
            return -2*np.matmul(X, np.transpose(X2)) + np.reshape(Xs, (-1,1)) + np.reshape(X2s, (1,-1))

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return np.sqrt(r2 + 1e-12)


    def build_prior_KL(self):
        """
        We return the KL for all latent funtions over all patients
        """
        KL = 0
        for p,ptid in enumerate(self.id_list):
            for q in np.arange(self.rank):  # q is the group id.
                for i in np.arange(self.num_latent_list[q]):
                    ### here is the mapping create a hashfunction to find latent fun of the ith latent fun in qth group of patient p. 
                    ### this is not neat at all but I don't know how to manage doing this without looping...
                    lat_id = p*self.num_latent_shared + np.sum(self.num_latent_list[:q],dtype = np.int64) + i #id of latent function
                    #if self.whiten_list[lat_id]:
                    #    if self.q_diag_list[lat_id]:
                    #        KL +=  kullback_leiblers.gauss_kl_white_diag(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], self.dim)
                    #    else:
                    #        KL += kullback_leiblers.gauss_kl_white(self.q_mu_list[lat_id],self.q_sqrt_list[lat_id], self.dim)
                    #else:
                        #compute with the kernel matrix of lat fun at qth kernel
                    #K = self.kern_list[q].K(self.Z[0]) + eye(self.num_inducing_list[lat_id]) * 1e-6
                    K = self.kern_list[q].K_precompd(self.dZ[0]) + eye(self.num_inducing_list[lat_id]) * 1e-6
                    #    if self.q_diag_list[lat_id]:
                    #        KL += kullback_leiblers.gauss_kl_diag(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], K, self.dim)
                    #    else:
                    KL += kullback_leiblers.gauss_kl(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], K, self.dim)
            if self.tsk:
                for d in np.arange(self.num_tasks):
                    lat_id = self.num_latent_shared * self.num_patients + p*int(self.num_tasks) + d #id of latent function
                    #if self.whiten_list[lat_id]:
                    #    if self.q_diag_list[lat_id]:
                    #        KL +=  kullback_leiblers.gauss_kl_white_diag(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], self.dim)
                    #    else:
                    #        KL += kullback_leiblers.gauss_kl_white(self.q_mu_list[lat_id],self.q_sqrt_list[lat_id], self.dim)
                    #else:
                        # compute with thet task specific kernel matrix of lat fun at d + p*num_tasks kernel
                    #K = self.tskern_list[d+p*int(self.num_tasks)].K(self.Z[0]) + eye(self.num_inducing_list[lat_id]) * 1e-6 ## compute with the ith kernel
                    K = self.tskern_list[d+p*int(self.num_tasks)].K_precompd(self.dZ[0]) + eye(self.num_inducing_list[lat_id]) * 1e-6 ## compute with the ith kernel
                    #    if self.q_diag_list[lat_id]:
                    #        KL += kullback_leiblers.gauss_kl_diag(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], K, self.dim)
                    #    else:
                    KL += kullback_leiblers.gauss_kl(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], K, self.dim)
        return KL #HS: this return was inside p loop!

    def build_likelihood(self):
        """
        Loglikelihood is the likelihood sum over all tasks
        the outer loop go through all the tasks and the inner loop goes through
        the latent function inside each task and calculate the loglike of this task
        as the weighted sum of loglike of latent function.
        """
        #Get prior KL.
        print(type(self.X[0]))
        sys.stdout.flush()
        KL,loglike  = self.build_prior_KL(),0
        #Get conditionals
        for p, ptid in enumerate(self.id_list):
            for d in np.arange(self.num_tasks):
                Fmean, Fvar, data_id = 0, 0, self.num_tasks * p + d #p-th patient at task d, we have this data_id to match the correct training data the latent function.
                lag_id = d -1
                if lag_id < 1: lag = 0
                else: lag = self.lagging[lag_id,0]
                for q in np.arange(self.rank):
                    for i in np.arange(self.num_latent_list[q]):
                        ## same idea as before
                        lat_id = p* self.num_latent_shared + np.sum(self.num_latent_list[:q],dtype = np.int64) + i
                        #if self.whiten_list[lat_id]:
                        #    fmean, fvar = conditionals.gaussian_gp_predict_whitened(self.X[data_id], self.Z[0],
                        #                    self.kern_list[q], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                        #else:
                        fmean, fvar = conditionals.gaussian_gp_predict(self.X[data_id] -lag, self.Z[0],
                                        self.kern_list[q], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1,
                                        precomp_dist=True, dX_Xnew=self.dZX[data_id],dX_X=self.dZ[0])
                        #add mean function to conditionals.
                        W_ij = self.W[d,np.sum(self.num_latent_list[:q],dtype = np.int64)+i]
                        #Fmean += (fmean + self.mean_function_list[lat_id](self.X[data_id]))*W_ij
                        Fmean += (fmean)*W_ij
                        Fvar += fvar * tf.square(W_ij) #the d-th task at ith latent
                if self.tsk:
                    ## note here task specific function is after the latent function
                    lat_id = self.num_latent_shared*self.num_patients + p*int(self.num_tasks) + d # specific latent function at task i for patient l.
                    #if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                    #    fmean, fvar = conditionals.gaussian_gp_predict_whitened(self.X[data_id], self.Z[0],
                    #                    self.tskern_list[d+p*int(self.num_tasks)], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                    #else:
                    fmean, fvar = conditionals.gaussian_gp_predict(self.X[data_id] - lag, self.Z[0],
                                    self.tskern_list[d+p*int(self.num_tasks)], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1,
                                    precomp_dist=True, dX_Xnew=self.dZX[data_id],dX_X=self.dZ[0])
                    #add mean function to conditionals.
                    #Fmean += (fmean + self.mean_function_list[lat_id](self.X[data_id]))*self.Kappa[d+p*int(self.num_tasks),0]
                    Fmean += (fmean)*self.Kappa[d+p*int(self.num_tasks),0]
                    Fvar += fvar * tf.square(self.Kappa[d+p*int(self.num_tasks),0])
                like_id = d   
                ve = self.likelihood[like_id].variational_expectations(Fmean, Fvar, self.Y[data_id])
                loglike += tf.reduce_sum(ve)     
        loglike -= KL
        return loglike

    def build_predict(self, Xnew, pt_id, task_ind):
            """
            We need to assume the task_ind starts from 0,
            We need to notice that the input are tf variables, so can't be indexed in the list
            """
            #print("***",pt_id, task_ind)
            Fmean,Fvar = 0,0
            for p,ptid in enumerate(self.id_list):
                switch_p = tf.cast(tf.equal(tf.to_int64(ptid), pt_id),dtype = tf.float64) #id matcher
                for q in np.arange(self.rank):
                    for i in np.arange(self.num_latent_list[q]):
                        lat_id = p*self.num_latent_shared+ np.sum(self.num_latent_list[:q],dtype = np.int64) + i
                        if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                            fmean, fvar = conditionals.gaussian_gp_predict_whitened(Xnew, self.Z[0],
                                            self.kern_list[q], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                        else:
                            fmean, fvar = conditionals.gaussian_gp_predict(Xnew, self.Z[0],
                                            self.kern_list[q], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                        W_ij = tf.gather(self.W,task_ind)[np.sum(self.num_latent_list[:q],dtype = np.int64) + i]
                        Fmean += (fmean + self.mean_function_list[lat_id](Xnew))*W_ij*switch_p
                        Fvar += fvar * tf.square(W_ij)*switch_p
                if self.tsk:
                    for d in np.arange(self.num_tasks):
                        lat_id = self.num_latent_shared*self.num_patients + p*int(self.num_tasks) + d # dth task for pth patient
                        if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                            fmean, fvar = conditionals.gaussian_gp_predict_whitened(Xnew, self.Z[0],
                                            self.tskern_list[d+p*int(self.num_tasks)], self.q_mu_list[lat_id],
                                            self.q_sqrt_list[lat_id],1)
                        else:
                            fmean, fvar = conditionals.gaussian_gp_predict(Xnew, self.Z[0],
                                            self.tskern_list[d+p*int(self.num_tasks)], self.q_mu_list[lat_id],
                                            self.q_sqrt_list[lat_id],1)
                        switch_d = tf.cast(tf.equal(tf.to_int64(d), task_ind),tf.float64)  #task matcher
                        W_ij = tf.gather(self.Kappa,d+p*np.int64(self.num_tasks))[0]*switch_d*switch_p
                        Fmean += (fmean + self.mean_function_list[lat_id](Xnew))*W_ij
                        Fvar += fvar * tf.square(W_ij)
            return Fmean, Fvar

    @AutoFlow(tf.placeholder(tf.float64, [None, None]), tf.placeholder(tf.int64,[]),tf.placeholder(tf.int64, []),tf.placeholder(tf.int64,[]))
    def predict_latent(self, Xnew, pt_id, group_id, ingroup_id):
        """
        Compute the posterior for one of the latent functions.
        """
        Fmean, Fvar = 0,0
        for p, ptid in enumerate(self.id_list):
            switch_p = tf.cast(tf.equal(ptid, pt_id),dtype = tf.float64)
            for q in np.arange(self.rank):
                switch_t = tf.cast(tf.equal(q, group_id),tf.float64)  #task matcher
                for i in np.arange(self.num_latent_list[q]):
                    lat_id = p*self.num_latent_funs + np.sum(self.num_latent_list[:q],dtype = np.int64) + i
                    if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                        fmean, fvar = conditionals.gaussian_gp_predict_whitened(Xnew, self.Z[0],
                                        self.kern_list[q], self.q_mu_list[lat_id],
                                        self.q_sqrt_list[lat_id], 1)
                    else:
                        fmean, fvar = conditionals.gaussian_gp_predict(Xnew, self.Z[0],
                                        self.kern_list[q], self.q_mu_list[lat_id],
                                        self.q_sqrt_list[lat_id], 1)
                    switch_i = tf.cast(tf.equal(i, ingroup_id),tf.float64)
                    Fmean += (fmean + self.mean_function_list[lat_id](Xnew))*switch_p*switch_t*switch_i
                    Fvar += fvar*switch_p*switch_t*switch_i
        return Fmean,Fvar

    @AutoFlow(tf.placeholder(tf.float64, [None,1]),tf.placeholder(tf.int64,[]),tf.placeholder(tf.int64, []))
    def predict_y(self, Xnew, pt_id, task_ind):
            """
            Compute the mean and variance of held-out data at the points Xnew
            """
            #print("in pred_y", pt_id, task_ind)
            pred_f_mean, pred_f_var = self.build_predict(Xnew,pt_id,task_ind)
            Ymean,Yvar =0,0
            for i in range(self.num_tasks):
                switch_t = tf.cast(tf.equal(task_ind,tf.to_int64(i)),tf.float64)
                ymean, yvar = self.likelihood[i].predict_mean_and_var(pred_f_mean, pred_f_var)  ## this is wrong , need to figure out how to index self.likelihood
                Ymean += ymean *switch_t
                Yvar += yvar*switch_t
            return Ymean, Yvar

    @AutoFlow(tf.placeholder(tf.float64),tf.placeholder(tf.int64,[]),tf.placeholder(tf.int64, []))
    def predict_f(self,Xnew,pt_id,task_ind):
        """
        We need to assume the task_ind starts from 0
        """
        return self.build_predict(Xnew,pt_id,task_ind)
