import numpy as np
import tensorflow as tf
from model import AutoFlow
#from model import ObjectiveWrapper
from scipy.optimize import minimize, OptimizeResult
from mean_functions import Zero
from param import ParamList,Param,Parameterized
from kernels import Kern
import transforms
import conditionals
import kullback_leiblers
from tf_hacks import eye
import pdb
import time
import sys
import GPflow
from kernels import Kern

class myLocalObjectiveWrapper(object):
    """
    A simple class to wrap the objective function in order to make it more robust.

    The previously seen state is cached so that we can easily access it if the
    model crashes.
    """
    _previous_x = None
    def __init__(self, objective):
        self._local_objective = objective
    def __call__(self, x, gx, feed_dict):
        f, g = self._local_objective(x, gx, feed_dict)
        g_is_fin = np.isfinite(g)
        if np.all(g_is_fin):
            self._previous_x = x # store the last known good value
            return f, g
        else:
            print("Warning: inf or nan in gradient: replacing with zeros")
            return f, np.where(g_is_fin, g, 0.)

class myGobalObjectiveWrapper(object):
    """
    A simple class to wrap the objective function in order to make it more robust.

    The previously seen state is cached so that we can easily access it if the
    model crashes.
    """
    _previous_x = None
    def __init__(self, objective):
        self._global_objective = objective
    def __call__(self, x, lx, feed_dict):
        f, g = self._global_objective(x, lx, feed_dict)
        g_is_fin = np.isfinite(g)
        if np.all(g_is_fin):
            self._previous_x = x # store the last known good value
            return f, g
        else:
            print("Warning: inf or nan in gradient: replacing with zeros")
            return f, np.where(g_is_fin, g, 0.)

class LMC_TSK(GPflow.model.Model):
    def __init__(self, num_tasks,rank,num_latent_list,kern_list, likelihood_list,
                  Z = None, TSK = False, TSKern_list=None, mean_function=Zero(),q_diag_list=None,
                  whiten_list = None,
                  needs_recompile = True, num_inducing_points = 10,
                  W = None, Kappa = None,locMin = 0, locMax = 5000,X = None, Y = None, lagging=None,lamda = 1):
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
        if whiten_list is None: 
            self.whiten_list = [False] * self.num_latent
        else: 
            self.whiten_list = [whiten_list] * self.num_latent
        if q_diag_list is None: 
            self.q_diag_list = [False] * self.num_latent
        else: 
            self.q_diag_list = [q_diag_list] * self.num_latent
        self.X, self.Y = X, Y
        """
        q_sqrt is a 3D tensor, each matrix within is a lower triangular square-root
        matrix of the covariance of q. num_tasks is the number of tasks , num_latent is the
        number of latent functions for each task. Assume here the number of latent function
        of each task is the same.
        """
        self.q_sqrt_list,self.q_mu_list,self.Z,self.kern_list, self.tskern_list,self.likelihood= \
            ParamList([]),ParamList([]),ParamList([]),ParamList([]),ParamList([]),ParamList([])

        self.num_inducing_list,self.num_tasks,self.rank,self.dim  = list(),num_tasks,rank,1
        """
        Here, our default assumption is dimemsion is 1
        """
        for i in range(self.rank):
            self.kern_list.append(kern_list[i])
         
        for q in range(self.num_tasks):
            self.likelihood.append(likelihood_list[q])

        """
        Assign kernel to each task specific function
        """
        if self.tsk:
            for d in range(self.num_tasks):
                self.tskern_list.append(TSKern_list[d])
                
        for q in range(self.num_latent):
            if Z is None: 
                self.Z.append(Param(np.linspace(locMin,locMax,num_inducing_points)[:,np.newaxis])) #10 is the default number of inducing points
                self.Z[q].fixed = True
            else: 
                self.Z.append(Param(Z[q]))
            self.num_inducing_list.append(self.Z[q].shape[0])   #initializing inducing points
            if self.q_diag_list[q]:
                self.q_sqrt_list.append(Param(np.ones((self.num_inducing_list[q], self.dim)),transforms.positive))
            else:
                self.q_sqrt_list.append(Param(np.array([np.eye(self.num_inducing_list[q]) for _ in range(self.dim)]).swapaxes(0,2)))
            
            self.q_mu_list.append(Param(np.zeros((self.num_inducing_list[q], self.dim)))) # dimension of Y is 1
        
        if W is None:
            self.W = Param(np.random.randn(self.num_tasks,np.sum(self.num_latent_list,dtype = np.int64)))  #dim n * m
        else:
            self.W = W
            
        if lagging is None:
            self.lagging = Param(np.random.randn(self.num_tasks,1), transforms.positive)
        else:
            self.lagging = lagging

        if self.tsk:
            if Kappa is None:
                self.Kappa = Param(0.05*np.random.randn(self.num_tasks,1))  #dim n * m]
            else:
                self.Kappa = Kappa
                
        self.lamda = lamda

    def build_prior_KL(self):
        """
        We return the KL for all latent funtions
        """
        KL = 0
        for q in np.arange(self.rank):  # q is the group id.
            for i in np.arange(self.num_latent_list[q]):
                lat_id = np.sum(self.num_latent_list[:q],dtype = np.int64) + i #id of latent function
                if self.whiten_list[lat_id]:
                    if self.q_diag_list[lat_id]:
                        KL +=  kullback_leiblers.gauss_kl_white_diag(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], self.dim)
                    else:
                        KL += kullback_leiblers.gauss_kl_white(self.q_mu_list[lat_id],self.q_sqrt_list[lat_id], self.dim)
                else:
                    K = self.kern_list[q].K(self.Z[lat_id]) + eye(self.num_inducing_list[lat_id]) * 1e-6 
                    if self.q_diag_list[lat_id]:
                        KL += kullback_leiblers.gauss_kl_diag(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], K, self.dim)
                    else:
                        KL += kullback_leiblers.gauss_kl(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], K, self.dim)
        if self.tsk:
            for d in np.arange(self.num_tasks):
                lat_id = np.sum(self.num_latent_list,dtype = np.int64) + d#id of latent function
                if self.whiten_list[lat_id]:
                    if self.q_diag_list[lat_id]:
                        KL +=  kullback_leiblers.gauss_kl_white_diag(self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], self.dim)
                    else:
                        KL += kullback_leiblers.gauss_kl_white(self.q_mu_list[lat_id],self.q_sqrt_list[lat_id], self.dim)
                else:
                    K = self.tskern_list[d].K(self.Z[lat_id]) + eye(self.num_inducing_list[lat_id]) * 1e-6 ## compute with the ith kernel
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
    
        self.Xt = [tf.placeholder(dtype = tf.float64, shape = [None,1]) for _ in np.arange(self.num_tasks)]
        self.Yt = [tf.placeholder(dtype = tf.float64, shape = [None,1]) for _ in np.arange(self.num_tasks)]
        #Get conditionals, is this correct???
        for d in np.arange(self.num_tasks):
            ve,Fmean,Fvar = 0,0,0
            lag = self.lagging[d,0]
            for q in np.arange(self.rank):
                for i in np.arange(self.num_latent_list[q]):
                    lat_id = np.sum(self.num_latent_list[:q],dtype = np.int64) + i
                    if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                        fmean, fvar = conditionals.gaussian_gp_predict_whitened(self.Xt[d]-lag, self.Z[lat_id],
                                        self.kern_list[q], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                    else:
                        fmean, fvar = conditionals.gaussian_gp_predict(self.Xt[d]-lag,self.Z[lat_id],
                                        self.kern_list[q], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                    Fmean += (fmean + self.mean_function_list[lat_id](self.Xt[d]))*self.W[d,lat_id]
                    Fvar += fvar * tf.square(self.W[d,lat_id])
            if self.tsk:
                lat_id = np.sum(self.num_latent_list,dtype = np.int64) + d
                if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                    fmean, fvar = conditionals.gaussian_gp_predict_whitened(self.Xt[d], self.Z[lat_id],
                                    self.tskern_list[d], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                else:
                    fmean, fvar = conditionals.gaussian_gp_predict(self.Xt[d], self.Z[lat_id],
                                    self.tskern_list[d], self.q_mu_list[lat_id], self.q_sqrt_list[lat_id], 1)
                Fmean += (fmean + self.mean_function_list[lat_id](self.Xt[d]))*self.Kappa[d,0]
                Fvar += fvar * tf.square(self.Kappa[d,0])               
            ve = self.likelihood[d].variational_expectations(Fmean, Fvar, self.Yt[d])
            loglike += tf.reduce_sum(ve)  

        """add time to event likelihood to the loglike"""    
        return loglike - KL - self.lamda *tf.sqrt(tf.reduce_sum(tf.square(self.Kappa)))/self.num_patients
    
    def build_predict(self,Xnew,task_ind):
            """
                We need to assume the task_ind starts from 0
            """
            Fmean,Fvar = 0,0
            for d in np.arange(self.num_tasks):
                task_switch = tf.cast(tf.equal(tf.to_int64(d), task_ind),tf.float64)
                lag = self.lagging[d,0]
                for q in np.arange(self.rank):
                    for i in np.arange(self.num_latent_list[q]):
                        lat_id = np.sum(self.num_latent_list[:q],dtype = np.int64) + i
                        if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                            fmean, fvar = conditionals.gaussian_gp_predict_whitened(Xnew-lag, self.Z[lat_id],
                                            self.kern_list[q], self.q_mu_list[lat_id],
                                             self.q_sqrt_list[lat_id], 1)
                        else:
                            fmean, fvar = conditionals.gaussian_gp_predict(Xnew-lag, self.Z[lat_id],
                                            self.kern_list[q], self.q_mu_list[lat_id],
                                             self.q_sqrt_list[lat_id],1)
                        W_ij = tf.gather(self.W,task_ind)[lat_id]*task_switch
                        Fmean += (fmean + self.mean_function_list[lat_id](Xnew))*W_ij
                        Fvar += fvar * tf.square(W_ij)
                if self.tsk:
                    lat_id = np.sum(self.num_latent_list,dtype = np.int64) + d
                    if self.whiten_list[lat_id]:  # need to compute fmean and fvar by the weights
                        fmean, fvar = conditionals.gaussian_gp_predict_whitened(Xnew, self.Z[lat_id],
                                        self.tskern_list[d], self.q_mu_list[lat_id],
                                         self.q_sqrt_list[lat_id], 1)
                    else:
                        fmean, fvar = conditionals.gaussian_gp_predict(Xnew, self.Z[lat_id],
                                        self.tskern_list[d], self.q_mu_list[lat_id],
                                         self.q_sqrt_list[lat_id], 1)
                    #switch = tf.cast(tf.equal(tf.to_int32(d), task_ind),tf.float32)
                    W_ij = tf.gather(self.Kappa,d)[0]*task_switch
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
    
    
    @AutoFlow(tf.placeholder(tf.float64,[None,1]),tf.placeholder(tf.int64, []))    
    def predict_task_latent(self,Xnew,task_id):
        Fmean,Fvar = 0,0
        for i in xrange(self.num_tasks):
            switch = tf.cast(tf.equal(np.int64(i),tf.to_int64(task_id)),tf.float64)
            lat_id = np.sum(self.num_latent_list) + i
            if self.whiten_list[lat_id]:
                fmean,fvar = conditionals.gaussian_gp_predict_whitened(Xnew, self.Z[lat_id],
                                    self.tskern_list[i], self.q_mu_list[lat_id],
                                     self.q_sqrt_list[lat_id],1)
            else:
                fmean, fvar = conditionals.gaussian_gp_predict(Xnew, self.Z[lat_id],
                                    self.tskern_list[i], self.q_mu_list[lat_id],
                                     self.q_sqrt_list[lat_id], 1)
            Fmean += fmean*switch
            Fvar += fvar*switch
        return Fmean, Fvar


    
    def _compile(self):
        """
        compile the tensorflow function "self._objective"
        """
        #self._unfix_all_params()
        print('compile all',self.get_free_state().shape)
        free_vars_copy = self.get_free_state()
        self.params = tf.placeholder(shape = self.get_free_state().shape,dtype = tf.float32)
        self._free_vars32 = tf.Variable(self.params,name = 'all_variables',dtype =tf.float32)
        self._free_vars = tf.cast(self._free_vars32,dtype = tf.float64)
        self.make_tf_array(self._free_vars,0)

        self.local_indices = []
        self.tsk_indices = []
        for param in self.q_mu_list: self.local_indices.extend(param.free_array_ind)
        for param in self.q_sqrt_list: self.local_indices.extend(param.free_array_ind)
        #for z in self.Z: z.fixed = True
        if self.tsk:
            for kern in self.tskern_list:
                if not kern.lengthscales.fixed:
                    self.local_indices.extend(kern.lengthscales.free_array_ind)
                    self.tsk_indices.extend(kern.lengthscales.free_array_ind)
        self.local_indices = np.array(sorted(self.local_indices))
        if self.tsk:
            self.tsk_indices = np.array(sorted(self.tsk_indices))

        self.global_indices = []
        if not self.W.fixed:
            self.global_indices.extend(self.W.free_array_ind)
        if self.tsk and (not self.Kappa.fixed):
            self.global_indices.extend(self.Kappa.free_array_ind)
        for d in range(self.num_tasks):
            if not self.likelihood[d].sorted_params[0].fixed:
                self.global_indices.extend(self.likelihood[d].sorted_params[0].free_array_ind)
        for kern in self.kern_list:
            if not kern.lengthscales.fixed:
                self.global_indices.extend(kern.lengthscales.free_array_ind)
        if not self.lagging.fixed:
            self.global_indices.extend(self.lagging.free_array_ind)
        self.global_indices = np.array(sorted(self.global_indices))

        print(len(self.global_indices), len(self.local_indices))


        #print(len(self.global_indices), len(self.local_indices))

        self._nfree_params = free_vars_copy.shape
        tf_local_indices = tf.constant(self.local_indices)
        tf_global_indices = tf.constant(self.global_indices)
        self.local_mask = tf.cast(tf.sparse_to_dense(tf_local_indices, self._nfree_params, 1), tf.bool)
        self.global_mask = tf.cast(tf.sparse_to_dense(tf_global_indices, self._nfree_params, 1), tf.bool)

        #tf_zeros = tf.zeros(free_vars_copy.shape, dtype = tf.float64)

        with self.tf_mode():
            self._f = self.build_likelihood() + self.build_prior()
            #self.g_global = tf.gradients(self.f, self._free_vars_global)
            self._grads = tf.gradients(self._f, self._free_vars)[0]
            #self._g_global = tf.select(self.global_mask, grads[0], tf_zeros)
            #self._g_local = tf.select(self.local_mask, grads[0], tf_zeros)
            #self.g_local = tf.gather(grads[0], tf_local_indices)

        def obj(x, gx, feed_dict):
            in_dict = feed_dict.copy()
            free_vars = np.zeros(self._nfree_params)
            free_vars[self.global_indices] = gx
            free_vars[self.local_indices] = x
            in_dict.update({self._free_vars:free_vars})
            fval, jac = self._session.run([self._f, self._grads], feed_dict=in_dict)
            return -fval, -jac[self.local_indices]
        self._local_objective = obj
        #self._local_obj = myLocalObjectiveWrapper(self._local_objective)

        def g_obj(x, lx, feed_dict):
            in_dict = feed_dict.copy()
            free_vars = np.zeros(self._nfree_params)
            free_vars[self.global_indices] = x
            free_vars[self.local_indices] = lx
            in_dict.update({self._free_vars:free_vars})
            fval, jac = self._session.run([self._f, self._grads], feed_dict=in_dict)
            return fval, jac[self.global_indices]
        self._global_objective = g_obj

        return free_vars_copy

    def _sgd(self, X_list, Y_list, max_iters, local_iters, method, options):
        """
        X,Y is a list of list X[i] contains all the training data for one patient, X[i] is a list of np.array
        """
        free_vars_copy = self._compile()

        free_vars_global, local_vars, _minusF, log_F =[None], [None]*len(X_list), [None]*len(X_list), list()
        step, previous_objective, _continue = 0, 0, True
        tsk_vars = [None]*len(X_list)

        self.tsk_in_local_indices = np.where([np.where(self.local_indices==x)[0][0] for x in self.tsk_indices])
        for order in np.arange(len(X_list)):
            local_vars[order] = free_vars_copy.astype(np.float64)[self.local_indices].copy()
            if self.tsk:
                tsk_vars[order] = free_vars_copy.astype(np.float64)[self.tsk_indices].copy()
        global_vars = free_vars_copy.astype(np.float32)[self.global_indices].copy()

        if method == 'adam':
            # adam stuff
            self._adam_alpha = options['alpha']
            self._adam_beta1 = options['beta1']
            self._adam_beta2 = options['beta2']
            self._adam_eps = 1e-8
            self._adam_m = np.zeros(global_vars.shape)
            self._adam_v = np.zeros(global_vars.shape)
        else:
            self._adagrad_alpha = options['alpha']
            self._adagrad_v = np.zeros(global_vars.shape)

        init = tf.initialize_all_variables()

        while _continue and step < max_iters:
            #permutation = np.random.permutation(np.arange(len(X_list)))
            order = np.random.choice(len(X_list), 1)
            #for order, X, Y in zip(permutation,np.array(X_list)[permutation],np.array(Y_list)[permutation]):

            """set free vars to previous value"""
            free_vars = np.zeros(self._nfree_params)
            free_vars[self.local_indices] = free_vars_copy.astype(np.float64)[self.local_indices].copy() # re-init local values
            #free_vars[self.local_indices] = local_vars[order].copy() # take previous values
            #if self.tsk:
            #    free_vars[self.tsk_indices] = tsk_vars[order].copy() # take previous values for tskern params
            free_vars[self.global_indices] = global_vars.copy()
            #self._session.run(init,)

            try:
                feed_dict = {self.params: free_vars}
                feed_1,feed_2 = {i:d for i,d in zip(self.Xt, X_list[order])},{i:d for i,d in zip(self.Yt,Y_list[order])}
                feed_dict.update(feed_1);feed_dict.update(feed_2)

                t0 = time.time()
                opt_res, F,final_local_vars = self.local_optimize_np(local_vars[order].astype(np.float64).copy(),
                                                                     global_vars.astype(np.float64).copy(),
                                                                     feed_dict, max_iters=local_iters)#, method='CG')
                t1 = time.time()

                #var = np.zeros(free_vars_copy.shape)
                #var[self.global_indices] = global_vars
                #var[self.local_indices] = final_local_vars.astype(np.float32).copy()
                #print ('patient %d: lkh = %f, opt success = %r, time = %f\n' %(order, -F, opt_res.success, t1-t0))


                # copy local variables after update
                local_vars[order] = final_local_vars.astype(np.float32).copy()#var[self.local_indices].copy()
                if self.tsk:
                    tsk_vars[order] = final_local_vars[self.tsk_in_local_indices].astype(np.float32).copy()#var[self.tsk_indices].copy()
                #self.set_state(var)

                free_vars = np.zeros(self._nfree_params)
                free_vars[self.local_indices] = local_vars[order].copy()
                free_vars[self.global_indices] = global_vars.copy()

                step += 1
                previous_global = global_vars.copy()
                t0 = time.time()
                Fg, updated_global_vars = self.global_optimize_np(global_vars.astype(np.float64).copy(),
                                                              local_vars[order].astype(np.float64).copy(),
                                                              feed_dict, step, method)
                t1 = time.time()
                #print('after',updated_global_vars)
                global_vars = updated_global_vars.astype(np.float32).copy()
                """
                    TODO: return global w and kernel params.
                """
                var = np.zeros(self._nfree_params)
                var[self.global_indices] = updated_global_vars.astype(np.float32).copy()
                var[self.local_indices] = local_vars[order].astype(np.float32).copy()
                self.set_state(var)

                gvar_change = np.sum(np.fabs((previous_global - var[self.global_indices])/previous_global))
                if step%50 == 0:
                    print('step %d, change %f -- time %f' %(step, gvar_change, t1-t0))
                if gvar_change < 10**(-5):
                    _continue = False

            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, setting model with most recent state.")
                self.set_state(self._session.run(self._free_vars))
                return None

        ### final update of all local variables
        permutation = np.random.permutation(np.arange(len(X_list)))
        for order, X, Y in zip(permutation,np.array(X_list)[permutation],np.array(Y_list)[permutation]):

            """set free vars to previous value"""
            free_vars = np.zeros(free_vars_copy.shape)
            free_vars[self.local_indices] = free_vars_copy.astype(np.float64)[self.local_indices].copy() # re-init local values
            #free_vars[self.local_indices] = local_vars[order].copy() # take previous values
            #if self.tsk:
            #    free_vars[self.tsk_indices] = tsk_vars[order].copy() # take previous values for tskern params
            free_vars[self.global_indices] = global_vars.copy()
            self._session.run(init,feed_dict = {self.params: free_vars})

            feed_dict = dict()
            feed_1,feed_2 = {i:d for i,d in zip(self.Xt, X)},{i:d for i,d in zip(self.Yt,Y)}
            feed_dict.update(feed_1);feed_dict.update(feed_2)

            opt_res, F,final_local_vars = self.local_optimize_np(local_vars[order].astype(np.float64).copy(),
                                                                 global_vars.astype(np.float64).copy(),
                                                                 feed_dict, max_iters=1000)#, method='CG')
                                                                 #feed_dict, max_iters=local_iters)
            var = np.zeros(free_vars_copy.shape)
            var[self.global_indices] = global_vars
            var[self.local_indices] = final_local_vars.astype(np.float32).copy()
            #print ('patient %d: lkh = %f, opt success = %r\n' %(order, -F, opt_res.success))
            # copy local variables after update
            local_vars[order] = var[self.local_indices].copy()

        return local_vars, global_vars

    def local_optimize_np(self, x0, global_x, feed_dict, method='L-BFGS-B', tol=None, callback=None, max_iters=100):

        options=dict(disp=False, maxiter=max_iters)

        obj = myLocalObjectiveWrapper(self._local_objective)

        try:#self._local_obj,
            result = minimize(fun=obj,
                        x0=x0,
                        method=method,
                        jac=True, # self._local_objective returns the objective and the jacobian.
                        tol=tol,
                        callback=callback,
                        options=options,
                        args=(global_x,feed_dict,))
        except (KeyboardInterrupt):
            print("Caught KeyboardInterrupt, setting model with most recent state.")
            self.set_state(obj._previous_x)
            return None

        #print("optimization terminated, setting model state")
        #self.set_state(result.x)
        return result, result.fun, result.x

    def global_optimize_np(self, x0, local_x, feed_dict, iter, method):

        obj = myGobalObjectiveWrapper(self._global_objective)

        fval, grad = obj(x0, local_x, feed_dict)

        if method == 'adam':
            self._adam_m = self._adam_beta1*self._adam_m + self._adam_beta1*grad
            self._adam_v = self._adam_beta2*self._adam_v + self._adam_beta2*(grad**2)
            alpha_t = self._adam_alpha*np.sqrt(1-self._adam_beta2**iter)/(1-self._adam_beta1**iter)
            x = x0 + alpha_t*self._adam_m/(np.sqrt(self._adam_v)+self._adam_eps)
        else:
            self._adagrad_v += grad**2
            x = x0 + self._adagrad_alpha*grad/(np.sqrt(self._adagrad_v)+1e-10)

        '''new_fval, grad = obj(x, local_x, feed_dict)
        if np.isfinite(new_fval):
            return new_fval, x
        else:
            print("global infinite")
            return fval, x0'''
        return fval, x

    def optimize(self, max_iters, local_iters, X_list, Y_list, method, options):

        local_vars, global_vars = self._sgd(X_list, Y_list, max_iters, local_iters,  method = method, options = options)
        return local_vars, global_vars, self.local_indices, self.global_indices


