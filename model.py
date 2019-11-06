import numpy as np
import tensorflow as tf
from keras.layers import Dropout

from tensorflow.keras.layers import Dense
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
    def __init__(self, n_gmm=2, latent_dim=3):

        super(DAGMM, self).__init__()

        # Encoder
        self.fc_1 = Dense(10, activation='tanh')
        self.fc_2 = Dense(2)

        # Decoder
        self.fc_3 = Dense(10, activation='tanh')
        self.fc_4 = Dense(274)

        # GMM
        self.fc_5 = Dense(10, activation='tanh')
        # self.do_1 = Dropout(0.5)
        self.fc_6 = Dense(2, activation='softmax')

        self.encoded, self.decoded, self.estimation = None, None, None

    def call(self, inputs, training=None, mask=None):
        print("calling...")
        if training == True:
            self.forward(inputs)
            phi, mu, cov = self.compute_gmm_params(self.encoded, self.estimation)
        else:
            self.forward(inputs)
