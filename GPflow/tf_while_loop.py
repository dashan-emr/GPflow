import GPflow
import tensorflow as tf
import numpy as np
from model import AutoFlow


class SPLFM_many(GPflow.model.Model):
    def __init__(self, data, Z, num_latent):
        self.data = data
        self.num_patients = data.shape[0]
        self.max_num_data = data.shape[1]
        self.num_outputs = data.shape[2] - 1
        self.num_latent = num_latent
        
        self.Z = Z  # GPflow.param.Param?
        self.num_inducing = self.Z.shape[0]
        GPflow.model.Model.__init__(self,name = 'ManyLMC')
        W0 = np.random.randn(self.num_latent, self.num_outputs) 

        # init W, kappa
        self.W = GPflow.param.Param(W0)
        self.kappa = GPflow.param.Param(np.abs(np.random.randn(self.num_outputs,)),GPflow.transforms.positive)

        # init variational dists
        # for latent functions:
        self.q_mu = GPflow.param.Param(np.zeros((self.num_patients, self.num_inducing, self.num_latent)))
        self.q_sqrt = GPflow.param.Param(np.ones((self.num_patients, self.num_inducing, self.num_latent)))

        # for independent-functions:
        self.q_mu_op = GPflow.param.Param(np.zeros((self.num_patients, self.num_inducing, self.num_outputs)))
        self.q_sqrt_op = GPflow.param.Param(np.ones((self.num_patients, self.num_inducing, self.num_outputs)))

        # init kernels
        self.latent_kern = GPflow.kernels.Matern32(1,1)
        self.latent_kern.variance.fixed = True
        self.output_kern = GPflow.kernels.Matern32(1,1)
        self.output_kern.variance.fixed = True

        self.likelihood = GPflow.likelihoods.Gaussian()


    def _compile(self, optimizer=None):
        """
        compile the tensorflow function "self._objective"
        """
        # Make float32 hack
        self.need_gradient = False
        float32_hack = False
        if optimizer is not None:
            if tf.float64 not in optimizer._valid_dtypes() and tf.float32 in optimizer._valid_dtypes():
                print("Using float32 hack for Tensorflow optimizers...")
                float32_hack = True

        self._free_vars = tf.Variable(self.get_free_state())

        ## why not convert to tf.float64 directly
        #start = time.time()
        if float32_hack:
            self._free_vars32 = tf.Variable(self.get_free_state().astype(np.float32))
            self._free_vars = tf.cast(self._free_vars32, tf.float64)
        #print ("running time for building free vars in float32_hack is %.3f " %(time.time() - start))
        with tf.name_scope('make_params'):
            self.make_tf_array(self._free_vars)

        init = tf.initialize_all_variables()
        self._session.run(init)
        start = time.time()
        with self.tf_mode():
            #f = self.build_likelihood() + self.build_prior()
            f =  self.build_likelihood() 
            #if self.need_gradient:
            g, = tf.gradients(f, self._free_vars)
        print ("running time for building objective and gradient is: %.3f seconds" % (time.time() - start))   
        self._minusF = tf.neg( f, name = 'objective' )
        #print (self._session.run(g))
        #pdb.set_trace()
        print ("minusF initial is %f" %(self._session.run(self._minusF)))
        #pdb.set_trace()
        #sys.stdout()
        if self.need_gradient:
        	self._minusG = tf.neg( g, name = 'grad_objective' )
        # The optimiser needs to be part of the computational graph, and needs
        # to be initialised before tf.initialise_all_variables() is called.
        if optimizer is None:
            opt_step = None
        else:
            if float32_hack:
                print ("building optimizer ops...")
                start = time.time()
                opt_step = optimizer.minimize(tf.cast(self._minusF, tf.float32), var_list=[self._free_vars32])
                print ("running time for building optimzier is %.3f" %(time.time() - start))
            else:
                print ("building optimizer ops...")
                start = time.time()
                opt_step = optimizer.minimize(self._minusF, var_list=[self._free_vars])
                print ("running time for building optimzier is %.3f" %(time.time() - start))
            sys.stdout.flush()

        #build tensorflow functions for computing the likelihood and predictions
        print("compiling tensorflow function...")
        sys.stdout.flush()
        if self.need_gradient:
            def obj(x):
                return self._session.run([self._minusF, self._minusG], feed_dict={self._free_vars: x})
        else:
            def obj(x):
                return self._session.run(self._minusF, feed_dict={self._free_vars: x})
        self._objective = obj
        print("done")
        sys.stdout.flush()
        self._needs_recompile = False

        return opt_step



    def _optimize_tf(self, method, callback, max_iters, calc_feed_dict):
        """
        Optimize the model using a tensorflow optimizer. see self.optimize()
        """
        self.need_gradient = False
        opt_step = self._compile(optimizer=method)
        previosu_f = 0
        #tf.scalar_summary('objectiveF', self._minusF)
        #summary_op = tf.merge_all_summaries()
        #summary_writer = tf.train.SummaryWriter(logdir='./logdir', graph=self._session.graph)
        #print ("ready to write graph")
        try:
            iteration = 0
            while iteration < max_iters:
                if calc_feed_dict is None:
                    feed_dict = {}
                else:
                    feed_dict = calc_feed_dict()
                #start = time.time()    
                try:
                	_,f,free_vars = self._session.run([opt_step,self._minusF,self._free_vars32], feed_dict=feed_dict)
                	print "\riteration: %d , likelihood: %.3f " %(iteration, f), 
                except error:
                	"""
                		if the likelihood goes to nan or inf,
                		re initialize variables.
                	"""
                	print 're initializing variables'
                	self._free_vars32  = tf.assign(self._free_vars32,self.get_free_state()) 
                #tf.Print(tmp_array,[tmp_array])
                #print ("running time for this iteratoin is: %.3f seconds" % (time.time() - start))
                #summary_writer.add_summary(summary, global_step=None)
                if np.abs(previosu_f - f) < 10**(-2):
                    break
                else:
                    previosu_f = f
                if callback is not None:
                    callback(self._session.run(self._free_vars))
                iteration += 1
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, setting model with most recent state.")
            self.set_state(self._session.run(self._free_vars))
            return None
        final_x = self._session.run(self._free_vars)
        self.set_state(final_x)
        if self.need_gradient:
            fun, jac = self._objective(final_x)
            r  = OptimizeResult(x=final_x,
                           success=True,
                           message="Finished iterations.",
                           fun=fun,
                           jac=jac,
                           status="Finished iterations.")
        else:
            fun = self._objective(final_x)
            r  = OptimizeResult(x=final_x,
                           success=True,
                           message="Finished iterations.",
                           fun=fun,
                           status="Finished iterations.")
       
        return r    


    def build_likelihood(self):
        result = tf.zeros((), tf.float64)
        counter = tf.zeros((), tf.int32)

        def cond(r, i):
            return tf.not_equal(i, self.num_patients)

        def body(r, i):
            # this function computes the likelihood for a single patient.
            q_mu_i = tf.slice(self.q_mu, tf.pack([i, 0, 0]), tf.pack([1, -1, -1]))[0, :,:]
            q_sqrt_i = tf.slice(self.q_sqrt, tf.pack([i, 0, 0]), tf.pack([1, -1, -1]))[0, :,:]

            q_mu_op_i = tf.slice(self.q_mu_op, tf.pack([i, 0, 0]), tf.pack([1, -1, -1]))[0, :,:]
            q_sqrt_op_i = tf.slice(self.q_sqrt_op, tf.pack([i, 0, 0]), tf.pack([1, -1, -1]))[0, :,:]

            data_i = tf.slice(self.data, tf.pack([i, 0, 0]), tf.pack([1, -1, -1]))[0, :,:]
    
            X_i = data_i[:, :1]
            Y_i = data_i[:, 1:]
            #ind = tf.logical_not(tf.is_nan(X_i))

            # get non-nan X, Y. the data is followed by a series of nans for padding.
            valid = tf.argmax(tf.cast(tf.is_nan(tf.reshape(X_i, [-1])), tf.int32), 0)
            X_i = tf.slice(X_i,begin = tf.zeros([2,], tf.int64),size = tf.pack([valid,-1]))  #n*1
            Y_i = tf.slice(Y_i, begin = tf.zeros([2,], tf.int64), size = tf.pack([valid,-1])) #n*m (m is number of outputs)

            # build the Gaussian dist on outputs, starting with latent GPs:
            mu, var = GPflow.conditionals.conditional(X_i, self.Z,
                                          self.latent_kern,
                                          q_mu_i,
                                          self.num_latent,
                                          full_cov=False,
                                          q_sqrt= q_sqrt_i,
                                          whiten=True)
            mu = tf.matmul(mu, self.W) #dim mu = #Points * #outputs
            var = tf.matmul(var, tf.square(self.W))

            # get extra output-independent variance
            
            mu_op, var_op = GPflow.conditionals.conditional(X_i, self.Z,
                                                self.output_kern,
                                                q_mu_op_i,
                                                self.num_outputs,
                                                full_cov=False,
                                                q_sqrt=q_sqrt_op_i,
                                                whiten=True)  #Points * #Outputs 
           
            # compute overall op distributions:
            mu = mu + mu_op * self.kappa 
            var = var + var_op * tf.square(self.kappa)
            

            KL = GPflow.kullback_leiblers.gauss_kl_white_diag(q_mu_i, q_sqrt_i, self.num_latent)
            KL += GPflow.kullback_leiblers.gauss_kl_white_diag(q_mu_op_i, q_sqrt_op_i, self.num_outputs)

            likelihood = tf.reduce_sum(self.likelihood.variational_expectations(mu,var,Y_i)) - KL
            return r + likelihood, i+1


        result, counter = tf.while_loop(cond, body, [result, counter])
        return result


    def build_predict(self,Xnew,ptid):
    	q_mu = tf.slice(self.q_mu,begin = tf.pack([ptid,0,0]), size = tf.pack([1,-1,-1]))[0,:,:]
    	q_sqrt = tf.slice(self.q_sqrt, begin = tf.pack([ptid,0,0]), size = tf.pack([1,-1,-1]))[0,:,:]

    	q_mu_op = tf.slice(self.q_mu_op, begin = tf.pack([ptid, 0, 0]), size = tf.pack([1, -1, -1]))[0,:,:]
    	q_sqrt_op = tf.slice(self.q_sqrt_op, begin = tf.pack([ptid, 0, 0]), size = tf.pack([1, -1, -1]))[0,:,:]

    	mu, var = GPflow.conditionals.conditional(Xnew, self.Z,
                                          self.latent_kern,
                                          q_mu,
                                          self.num_latent,
                                          full_cov=False,
                                          q_sqrt= q_sqrt,
                                          whiten=True)

    	mu = tf.matmul(mu, self.W) #dim mu = #Points * #outputs
        var = tf.matmul(var, tf.square(self.W))
    	mu_op, var_op = GPflow.conditionals.conditional(Xnew, self.Z,
                                                self.output_kern,
                                                q_mu_op,
                                                self.num_outputs,
                                                full_cov=False,
                                                q_sqrt=q_sqrt_op,
                                                whiten=True)  #Points * #Outputs 

    	mu = mu + mu_op * self.kappa 
        var = var + var_op * tf.square(self.kappa)

        return mu, var

    @AutoFlow(tf.placeholder(tf.float64, [None, 1]), tf.placeholder(tf.int32,None))
    def predict_y(self,Xnew,ptid):
    	mu, var = self.build_predict(Xnew,ptid)
    	return self.likelihood.predict_mean_and_var(mu, var)






