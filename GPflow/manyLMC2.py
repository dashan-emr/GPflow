import numpy as np
import tensorflow as tf
import param
from param import ParamList
import kernels
import lmc_tsk_shared
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import time
import transforms
import mean_functions 

class ManyLMC(object):
    
    def __init__(self, X, Y,id_list,num_tasks,rank,num_latent_list,kern_list,likelihood_list,num_inducing_points,
                  Z = None, TSK = False, TSKern_list=None, mean_function= mean_functions.Zero(),lagging = None,
                 q_diag_list=None,whiten_list = None,needs_recompile = True, locMin = 0 ,locMax = 5000):
            self.lmc_list = list()
            self.X, self.Y = list(),list() # invariant is a list of (m,1) array #event is a list of [1,1] array
            num_latent_shared = np.sum(num_latent_list)
            self.id_list = id_list
            self.W = param.Param(np.random.randn(num_tasks,num_latent_shared))  
            self.num_tasks = num_tasks
            self.lagging = lagging
            if lagging is None:
            	self.lagging = param.Param(np.fabs(np.random.randn(self.num_tasks,1)),transforms.positive)
            locMin = 1e100
            locMax = -1e100
            for index,ptid in enumerate(self.id_list):
                Xs = X[X[:,X.shape[1]-1] == float(ptid),:X.shape[1]-1] # extract data X at ptid and discard the id column
                Ys = Y[X[:,X.shape[1]-1] == float(ptid),:Y.shape[1]]
                X_tmp,Y_tmp = list(),list() #X_tmp contains training data for one patient
                for d in range(self.num_tasks):
                    X_tmp.append(Xs[Xs[:,Xs.shape[1]-1] == d,:Xs.shape[1]-1]) # discard the task column
                    Y_tmp.append(Ys[Xs[:,Xs.shape[1]-1] == d,:])
                    m1, M1 = np.min(X_tmp[-1]), np.max(X_tmp[-1])
                    if m1 < locMin:
                        locMin = m1
                    if M1 > locMax:
                        locMax = M1
                self.X.append(X_tmp);self.Y.append(Y_tmp)
            print('Min Z = %f, Max Z = %f, # Z = %d' %(locMin, locMax, num_inducing_points))
            self.w_log,self.length_log = [],[]
            ## pass all the parameters to the lmc_obj, the data matrix will be passed later
            self.lmc_obj = lmc_tsk_shared.LMC_TSK(num_tasks,rank,num_latent_list,kern_list,likelihood_list,
                    						Z = Z, TSK = TSK, TSKern_list=TSKern_list,
                    						mean_function=mean_function,q_diag_list=q_diag_list,
                    						whiten_list = whiten_list,needs_recompile = needs_recompile,
                    						num_inducing_points =num_inducing_points,W = self.W,locMin = locMin,locMax = locMax, 
                    						lagging=self.lagging,w_log = self.w_log, length_log = self.length_log)
            self.id_map = dict()
            for order,id in enumerate(self.id_list):
                self.id_map[id] = order

            self.num_patients = len(self.id_list)
                
    def optimize(self,max_iters, local_iters, method = 'adam', options={'alpha':0.01,'beta1':0.9,'beta2':0.999}):
        print ("optimizing...")
        self.lmc_obj.num_patients = self.num_patients
        self.local_vars, self.global_vars, self.local_indices, self.global_indices = \
            self.lmc_obj.optimize(max_iters = max_iters, local_iters = local_iters, X_list = self.X, Y_list = self.Y,
                                  method = method, options = options)
        print ("Done.")
    def predict_y(self,xnew,ptid,task_id):
        #print(ptid, task_id, self.id_map)
        order = self.id_map[ptid]
        #self.lmc_obj._unfix_all_params()
        free_vars = np.zeros((len(self.local_vars[order])+len(self.global_vars),))
        free_vars[self.local_indices] = self.local_vars[order].copy()
        free_vars[self.global_indices] = self.global_vars.copy()
        self.lmc_obj.set_state(free_vars)
        ymean,yvar = self.lmc_obj.predict_y(xnew,task_id)
        return ymean, yvar    

    def predict_latent(self,xnew, ptid, group_id, ingroup_id):
     	order = self.id_map[ptid]
     	free_vars = np.zeros((len(self.local_vars[order])+len(self.global_vars),))
        free_vars[self.local_indices] = self.local_vars[order].copy()
        free_vars[self.global_indices] = self.global_vars.copy()
        self.lmc_obj.set_state(free_vars)
        ymean,yvar = self.lmc_obj.predict_latent(xnew,group_id, ingroup_id)
        return ymean, yvar


    def predict_task_latent(self,xnew,ptid,task_id):
        order = self.id_map[ptid]
        free_vars = np.zeros((len(self.local_vars[order])+len(self.global_vars),))
        free_vars[self.local_indices] = self.local_vars[order].copy()
        free_vars[self.global_indices] = self.global_vars.copy()
        self.lmc_obj.set_state(free_vars)
        ymean,yvar = self.lmc_obj.predict_task_latent(xnew,task_id)
        return ymean, yvar

               
        