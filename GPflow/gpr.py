import tensorflow as tf
from .model import GPModel
from .param import Param
from .densities import multivariate_normal
from .mean_functions import Zero
import likelihoods
from tf_hacks import eye

class GPR(GPModel):
    def __init__(self, X, Y, kern, mean_function=Zero()):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects

        This is a vanilla implementation of GP regression with a Gaussian
        likelihood.  Multiple columns of Y are treated independently.
        """
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

    def build_likelihood(self):
        """
        Constuct a tensorflow function to compute the likelihood of a general GP model.

            \log p(Y, V | theta).

        """
        with tf.name_scope('kernel'):
            K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
            _ = tf.image_summary('k', tf.expand_dims(tf.expand_dims(tf.cast(K, tf.float32), 2), 0))
        L = tf.cholesky(K)
        with tf.name_scope('mean_function'):
            m = self.mean_function(self.X)

        with tf.name_scope('mvn_density'):
            log_lik = multivariate_normal(self.Y, m, L)

        return log_lik

    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X), lower=True)
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            fvar = tf.tile(tf.expand_dims(fvar, 2), tf.pack([1, 1, tf.shape(self.Y)[1]]))
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), reduction_indices=0)
            fvar = tf.tile(tf.reshape(fvar, (-1,1)), [1, self.Y.shape[1]])
        return fmean, fvar

 

