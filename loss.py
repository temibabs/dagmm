from tensorflow.python.keras.losses import Loss
import tensorflow as tf

class DAGMMLoss(Loss):
    def __init__(self, N=1024, lambda_1=0.1, lambda_2=0.005):
        super(DAGMMLoss, self).__init__()
        self.N = N
        self.lambda1 = lambda_1
        self.lambda2 = lambda_2

    def stage_loss(self, z, gamma, energy, cov, cov_diag):
        self.z = z
        self.gamma = gamma
        self.energy = energy
        self.cov = cov
        self.penalty = cov_diag

    def call(self, y_true, y_pred):
        N = y_true.shape[0]
        l2_norm = tf.norm(y_true - y_pred, axis=1)

        term_1 = tf.reduce_sum(l2_norm, axis=0)
        term_2 = self.lambda1 * tf.reduce_sum(self.energy) / N
        term_3 = self.lambda2 * self.penalty

        return term_1+ term_2 + term_3