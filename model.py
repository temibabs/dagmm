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
    def __init__(self, input_cols, num_enc, n_gmm=2, latent_dim=3):

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

    def call(self, inputs, training=None, mask=None):
        if training == True:
            temp = self.fc_1(inputs)
            temp = self.fc_2(temp)
            temp = self.fc_3(temp)
            enc = self.fc_4(temp)

            temp = self.fc_5(enc)
            temp = self.fc_6(temp)
            temp = self.fc_7(temp)
            dec = self.fc_8(temp)

            temp = self.fc_9(dec)
            z = self.do_1(temp)
            gamma = self.fc_10(z)

            return enc, dec, gamma

        else:
            self.forward(inputs)
