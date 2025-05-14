import tensorflow as tf
import tensorflow_probability as tfp

def build_bayesian_model(input_shape):
    tfpl = tfp.layers
    tfd = tfp.distributions

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tfpl.DenseVariational(64,
                              make_prior_fn=tfpl.default_mean_field_normal_fn(),
                              make_posterior_fn=tfpl.default_mean_field_normal_fn(),
                              kl_weight=1/input_shape[0],
                              activation='relu'),
        tfpl.DenseVariational(1,
                              make_prior_fn=tfpl.default_mean_field_normal_fn(),
                              make_posterior_fn=tfpl.default_mean_field_normal_fn(),
                              kl_weight=1/input_shape[0])
    ])
    return model
