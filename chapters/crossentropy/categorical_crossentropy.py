import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


tfd = tfp.distributions

x = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]], dtype=np.float32)
y_true = np.array([0, 1, 2], dtype=np.int32)  # Class indices


y_logits = tf.constant([[2.0, 1.0, 0.1],
                        [0.5, 2.2, 1.1],
                        [1.2, 0.7, 2.5]], dtype=tf.float32)


y_pred_dist = tfd.Categorical(logits=y_logits)

nll = -y_pred_dist.log_prob(y_true)

print("Cross-entropy losses for each sample:", nll.numpy())
print("Average cross-entropy loss:", tf.reduce_mean(nll).numpy())
