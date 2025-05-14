import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions

np.random.seed(42)
tf.random.set_seed(42)


x = np.linspace(-3, 3, 100)
y = 0.5 * x + 1 + np.random.normal(0, 0.2, size=x.shape)

x_train = x[..., np.newaxis].astype(np.float32)
y_train = y[..., np.newaxis].astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(
            loc=t[..., :1],
            scale=1e-3 + tf.nn.softplus(t[..., 1:])  
        )
    )
])


nll = lambda y, rv_y: -rv_y.log_prob(y)

model.compile(optimizer='adam', loss=nll)
model.fit(x_train, y_train, epochs=200, verbose=0)


x_test = np.linspace(-3, 3, 200)[..., np.newaxis].astype(np.float32)
y_pred_dist = model(x_test)
y_pred_mean = y_pred_dist.mean().numpy().squeeze()
y_pred_std = y_pred_dist.stddev().numpy().squeeze()


plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Noisy Data", alpha=0.6)
plt.plot(x_test.squeeze(), y_pred_mean, 'r', label="Mean Prediction")
plt.fill_between(x_test.squeeze(),
                 y_pred_mean - 2 * y_pred_std,
                 y_pred_mean + 2 * y_pred_std,
                 color='red', alpha=0.3, label="Uncertainty (±2 stddev)")
plt.title("Probabilistic Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
