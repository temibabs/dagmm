import sys

from tensorflow.python.keras.losses import Loss
from numpy import pi
import tensorflow as tf

import numpy as np

class DAGMMLoss(Loss):
    def __init__(self, N=1024, k=5, lambda_1=0.1, lambda_2=0.005):
        super(DAGMMLoss, self).__init__()
        self.N = N
        self.lambda1 = lambda_1
        self.lambda2 = lambda_2
        self.K = k

    def stage_loss(self, z, gamma):
        self.z = z
        self.gamma = gamma
        self.ready = True

    def call(self, y_true, y_pred):
        if not self.ready:
            raise NotImplemented

        phi, mu, cov = self.compute_parameters()
        self.compute_energy(phi, mu, cov)
        t1 = tf.reduce_mean(tf.square(y_true - y_pred))
        energy = self.compute_energy()
        t2 = energy * (self.lambda1 / self.N)

        self.ready = False
        return t1 + t2

    def compute_energy(self, phi, mu, cov):

        z_mu = tf.expand_dims(self.z, 1) - tf.expand_dims(self.mu, 0)

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        k, d, _ = tf.shape(cov)

        for i in range(self.K):
            cov_k = cov[i] + (tf.eye(d, dtype=tf.float64) * eps)
            cov_inverse.append(tf.expand_dims(tf.linalg.inv(cov_k), 0))

            det_cov.append(
                tf.expand_dims(
                    tf.reduce_prod(
                        tf.linalg.diag(
                        tf.linalg.cholesky(cov_k * (2*np.pi))
                    )
                ), axis=0)
            )
            cov_diag += tf.reduce_sum(1 / tf.linalg.diag(cov_k))

        cov_inverse = tf.concat(cov_inverse, axis=0)
        det_cov = tf.concat(det_cov, axis=0)

        exp_term_tmp = -0.5 * tf.reduce_sum(
            tf.reduce_sum(
                tf.expand_dims(z_mu, axis=-1) * tf.expand_dims(cov_inverse, axis=0),
                axis=-2
            ) * z_mu, axis=-1)
        max_val = tf.maximum()
        t = tf.reduce_sum(
            self.phi * (tf.math.exp(-0.5 * tf.transpose(z_mu) * (1 / self.cov) * z_mu))
            / tf.math.sqrt(tf.math.abs(2 * pi * self.cov)))
        energy = tf.math.log(t) * -1

        return energy

    def compute_parameters(self, z=None, gamma=None):
        sum_gamma = tf.reduce_sum(self.gamma, axis=0)

        # PHI
        phi = sum_gamma / self.N

        # mu = tf.reduce_sum(gamma*z, axis=0) / tf.reduce_sum(gamma, axis=0)
        mu = tf.reduce_sum(tf.expand_dims(self.gamma, axis=-1)
                           * tf.expand_dims(self.z, axis=1), axis=0) \
             / tf.expand_dims(sum_gamma, axis=-1)

        z_mu = tf.expand_dims(self.z, 1) - tf.expand_dims(mu, 0)
        z_mu_outer = tf.expand_dims(z_mu, -1) * tf.expand_dims(z_mu, -2)
        print('z_mu is {}'.format(z_mu.shape))
        cov = tf.reduce_sum(
            tf.expand_dims(tf.expand_dims(self.gamma, -1), -1) *
            z_mu_outer, axis=0) / tf.expand_dims(tf.expand_dims(sum_gamma, -1),
                                                 -1)
        self.phi, self.mu, self.cov = phi, mu, cov
        return phi, mu, cov
