import GPflow
import tensorflow as tf
import numpy as np
import unittest


class DumbModel(GPflow.model.Model):
    def __init__(self):
        GPflow.model.Model.__init__(self)
        self.a = GPflow.param.Param(3.)

    def build_likelihood(self):
        return -tf.square(self.a)


class NoArgsModel(DumbModel):

    @GPflow.model.AutoFlow()
    def function(self):
        return self.a


class TestNoArgs(unittest.TestCase):
    def setUp(self):
        self.m = NoArgsModel()
        self.m._compile()

    def test_return(self):
        self.failUnless(np.allclose(self.m.function(), 3.))


class TestAdd(unittest.TestCase):
    def setUp(self):

        x_tf = tf.placeholder(tf.float64)
        y_tf = tf.placeholder(tf.float64)

        class AddModel(DumbModel):
            @GPflow.model.AutoFlow(x_tf, y_tf)
            def add(self, x, y):
                return tf.add(x, y)

        self.m = AddModel()
        self.m._compile()
        rng = np.random.RandomState(0)
        self.x = rng.randn(10, 20)
        self.y = rng.randn(10, 20)

    def test_add(self):
        self.failUnless(np.allclose(self.x + self.y, self.m.add(self.x, self.y)))


class TestGPmodel(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        X, Y = rng.randn(2, 10, 1)
        self.m = GPflow.svgp.SVGP(X, Y, kern=GPflow.kernels.Matern32(1),
                                  likelihood=GPflow.likelihoods.StudentT(),
                                  Z=X[::2].copy())
        self.Xtest = np.random.randn(100, 1)
        self.Ytest = np.random.randn(100, 1)

    def test_predict_f(self):
        mu, var = self.m.predict_f(self.Xtest)

    def test_predict_y(self):
        mu, var = self.m.predict_y(self.Xtest)

    def test_predict_density(self):
        d = self.m.predict_density(self.Xtest, self.Ytest)


class TestFixAndPredict(unittest.TestCase):
    """
    Bug #54 says that if a model parameter is fixed  between calls to predict
    (an autoflow fn) then the second call fails. This test ensures replicates
    that and ensures that the bugfix remains in furure.
    """
    def setUp(self):
        rng = np.random.RandomState(0)
        X, Y = rng.randn(2, 10, 1)
        self.m = GPflow.svgp.SVGP(X, Y, kern=GPflow.kernels.Matern32(1),
                                  likelihood=GPflow.likelihoods.StudentT(),
                                  Z=X[::2].copy())
        self.Xtest = np.random.randn(100, 1)
        self.Ytest = np.random.randn(100, 1)

    def test(self):
        self.m._compile()
        self.m.kern.variance.fixed = True
        _, _ = self.m.predict_f(self.m.X)


if __name__ == "__main__":
    unittest.main()
