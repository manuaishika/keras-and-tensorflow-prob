import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import tensorflow_probability as tfp

def negative_log_likelihood(y_true, y_pred):
    return -tf.reduce_mean(tfp.distributions.Normal(loc=y_pred, scale=1).log_prob(y_true))

def train(model, X_train, Y_train, epochs=100):
    model.compile(optimizer='adam', loss=negative_log_likelihood, metrics=[MeanSquaredError()])
    history = model.fit(X_train, Y_train, epochs=epochs, verbose=1)
    return history
