from tensorflow.python.keras.losses import Loss
from numpy import pi
import tensorflow as tf


class DAGMMLoss(Loss):
    def __init__(self, z, N=118, k=10, lambda_1=0.1, lambda_2=0.005):
        super(DAGMMLoss, self).__init__()
        self.N = N
        self.lambda1 = lambda_1
        self.lambda2 = lambda_2
        self.K = k

    def call(self, y_true, y_pred):
        t1 = tf.reduce_mean(tf.square(y_true - y_pred))

        energy = self.compute_energy()
        t2 = energy * (self.lambda1 / self.N)

        return t1 + t2

    def compute_energy(self):
        phi, mu, self.cov = self.compute_parameters(self.z, self.gamma)
        g = self.z - mu
        t = tf.reduce_sum(phi * (tf.math.exp(-0.5 * tf.transpose(g) * (1/self.cov) * g))
                          / tf.math.sqrt(tf.math.abs(2*pi*self.cov)))
        energy = tf.math.log(t) * -1

        return energy

    def compute_parameters(self, z, gamma):
        for k in range(self.K):
            phi = tf.reduce_sum(self.gamma[k]/self.N)
            mu = tf.reduce_sum(self.gamma[k]*self.z)/\
                 tf.reduce_sum(self.gamma[k])
            t = (self.z-mu)
            cov = tf.reduce_sum(self.gamma[k]*t*tf.transpose(t))/\
                  tf.reduce_sum(self.gamma[k])

            return phi, mu, cov