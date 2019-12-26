# Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection in PyTorch

My attempt at reproducing the paper [Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://openreview.net/forum?id=BJJLHbb0-). Please Let me know if there are any bugs in my code. Thank you! =)

I implemented this on Python 3.7 using Tensorflow 2.0.0b1

### Dataset
KDDCup99 http://kdd.ics.uci.edu/databases/kddcup99/

### Some Test Results
Paper's Reported Results (averaged over 20 runs) : Precision : 0.9297, Recall : 0.9442, F-score : 0.9369

My Implementation (only one run) : Precision : 0.9677, Recall : 0.9538, F-score : 0.9607

### Visualizing the z-space:
<img src="https://github.com/danieltan07/dagmm/blob/master/z_space.png" width="50%"/>

### Some Implementation Details
Below are code snippets of the two main components of the model. More specifically, computing the gmm parameters and sample energy.

```python
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
```   
I added some epsilon on the diagonals of the covariance matrix, otherwise I get nan values during training.

I tried using `torch.potrf(cov_k).diag().prod()**2` to compute for the determinants, but for some reason I get errors after several epochs, so I used numpy's linalg to compute for the determinants instead.

```python
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

        ...
```
