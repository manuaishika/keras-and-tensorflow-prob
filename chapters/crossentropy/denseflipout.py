import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfpl = tfp.layers


np.random.seed(0)
x = np.linspace(-3, 3, 200).astype(np.float32)
y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)

x_train = x[..., np.newaxis]
y_train = y[..., np.newaxis]

model = tf.keras.Sequential([
    tfpl.DenseFlipout(64, activation='relu'),
    tfpl.DenseFlipout(64, activation='relu'),
    tfpl.DenseFlipout(1)
])

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

def output_distribution(x):
    loc = model(x)
    scale = 0.1  # fixed uncertainty
    return tfd.Normal(loc=loc, scale=scale)

inputs = tf.keras.Input(shape=(1,))
outputs = tfpl.DistributionLambda(make_distribution_fn=output_distribution)(inputs)
bnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)

bnn_model.compile(optimizer='adam', loss=negloglik)
bnn_model.fit(x_train, y_train, epochs=200, verbose=0)


x_test = np.linspace(-3, 3, 300, dtype=np.float32)[..., np.newaxis]
y_pred_dist = bnn_model(x_test)
y_mean = y_pred_dist.mean().numpy().squeeze()
y_std = y_pred_dist.stddev().numpy().squeeze()


plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b.', label="Training data")
plt.plot(x_test, y_mean, 'r-', label="Prediction mean")
plt.fill_between(x_test.squeeze(),
                 y_mean - 2 * y_std,
                 y_mean + 2 * y_std,
                 color='orange', alpha=0.3, label="±2 stddev")
plt.title("Bayesian Neural Network with DenseFlipout")
plt.legend()
plt.show()
