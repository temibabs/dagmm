import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout
import itertools
from utils import *


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.potrf(a, False)
        ctx.save_for_backward(l)
        return l

    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


class DAGMM(tf.keras.Model):
    """Residual Block."""
    def __init__(self, input_cols, num_enc):

        super(DAGMM, self).__init__()

        # Encoder
        self.fc_1 = Dense(60, activation='tanh')
        self.fc_2 = Dense(30, activation='tanh')
        self.fc_3 = Dense(10, activation='tanh')
        self.fc_4 = Dense(num_enc)

        # Decoder
        self.fc_5 = Dense(10, activation='tanh')
        self.fc_6 = Dense(30, activation='tanh')
        self.fc_7 = Dense(60, activation='tanh')
        self.fc_8 = Dense(input_cols)

        # GMM
        self.fc_9 = Dense(10, activation='tanh')
        self.do_1 = Dropout(0.5)
        self.fc_10 = Dense(10, activation='softmax')

        self.encoded, self.decoded, self.estimation = None, None, None

    def compute_parameters(self, z, gamma):
        N = gamma.shape[0]
        sum_gamma = tf.reduce_sum(gamma, axis=0)
        self.phi = sum_gamma / N

        self.mu = tf.reduce_sum(tf.expand_dims(gamma, axis=-1)
                                * tf.expand_dims(z, axis=1),
                                axis=0) / tf.expand_dims(sum_gamma, axis=-1)
        z_mu = tf.expand_dims(z, axis=1) - tf.expand_dims(self.mu, axis=0)
        z_mu_outer = tf.expand_dims(z_mu, axis=-1) * tf.expand_dims(z_mu, axis=-2)
        self.cov = tf.reduce_sum(
            tf.expand_dims(
                tf.expand_dims(gamma, axis=-1),
                axis=-1) * z_mu_outer,
            axis=0) / tf.expand_dims(
            tf.expand_dims(sum_gamma, axis=-1),
            axis=-1)


    def compute_sample_energy(self):
        k, D, _ = self.cov.shape
        z_mu = tf.expand_dims(self.zc, axis=1) - tf.expand_dims(self.mu, axis=0)

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            a = self.cov[i]
            b = tf.multiply(tf.eye(D, dtype=tf.dtypes.float64), eps)
            cov_k = a + b
            cov_inverse.append(tf.expand_dims(tf.linalg.inv(cov_k), axis=0))

            with tf.device('/CPU:0'):
                det_cov.append(
                    tf.expand_dims(
                        tf.reduce_prod(
                            tf.linalg.diag(
                                tf.linalg.cholesky(
                                    np.linalg.det(cov_k*2*np.pi))))))
            cov_diag += cov_diag + tf.reduce_sum(1 / tf.linalg.tensor_diag_part(cov_k))

        cov_inverse = tf.concat(cov_inverse, axis=0)
        det_cov = tf.concat(det_cov, axis=0)

        exp_term_tmp = -0.5 * tf.reduce_sum(
            tf.reduce_sum(
                tf.expand_dims(z_mu,
                               axis=-1)
                * tf.expand_dims(cov_inverse,
                               axis=0),
                axis=-2),
            axis=-1)

        max_val = tf.clip_by_value(exp_term_tmp, clip_value_min=0.)
        tf.exp(-0.5 * tf.transpose(z_mu))

    def call(self, inputs, training=None, mask=None):

        model_output = {}
        if training == True:
            temp = self.fc_1(inputs)
            temp = self.fc_2(temp)
            temp = self.fc_3(temp)
            zc = self.fc_4(temp)    # low dimensional representation zc
            self.zc = zc

            temp = self.fc_5(zc)
            temp = self.fc_6(temp)
            temp = self.fc_7(temp)
            x_ = self.fc_8(temp)    # reconstructed x_
            self.x_ = x_

            temp = self.fc_9(x_)
            z = self.do_1(temp)
            gamma = self.fc_10(z)

            self.compute_parameters(zc, gamma)
            model_output['energy'] = self.compute_sample_energy()
            model_output['gamma'] = gamma
            model_output['z'] = z
            #model_output['dec'] = dec
            model_output['cov'] = self.cov
            #model_output['cov_diag'] = self.

            return model_output

        else:
            self.forward(inputs)
