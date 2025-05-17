import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from models.bayesian import build_bayesian_model

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Generate data with varying noise levels to demonstrate different uncertainty types
def generate_heteroscedastic_data(n=300):
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    
    # Create noise that increases with x (heteroscedastic)
    noise_scale = 0.1 + 0.3 * np.abs(X)
    noise = np.random.normal(0, noise_scale, size=X.shape)
    
    # Add a gap in the data to demonstrate epistemic uncertainty
    mask = np.logical_or(X < -0.5, X > 0.5)
    X_train = X[mask]
    
    Y = 2.0 * X + 1.0 + noise
    Y_train = Y[mask]
    
    return X_train.astype('float32'), Y_train.astype('float32'), X.astype('float32'), Y.astype('float32')

# Generate our data
X_train, y_train, X_full, y_full = generate_heteroscedastic_data()

# Build the model with learnable aleatoric uncertainty
def build_heteroscedastic_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tfp.layers.DenseVariational(64,
                                   make_prior_fn=tfp.layers.default_mean_field_normal_fn(),
                                   make_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                   kl_weight=1/input_shape[0],
                                   activation='relu'),
        tf.keras.layers.Dense(2)  # Output both mean and log_variance
    ])
    
    # Add distribution layer to output Normal distribution with learned variance
    distribution_model = tf.keras.Sequential([
        model,
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(
                loc=t[..., :1],
                scale=tf.math.softplus(t[..., 1:]) + 1e-6  # Ensure positive scale
            )
        )
    ])
    
    return distribution_model

# Build and compile the model
model = build_heteroscedastic_model(input_shape=(1,))

# Define negative log likelihood loss
def negative_log_likelihood(y_true, y_pred_distr):
    return -y_pred_distr.log_prob(y_true)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=negative_log_likelihood)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=1000,
                    verbose=1)

# Generate test points across the full range
X_test = np.linspace(-4, 4, 500).reshape(-1, 1).astype('float32')

# Multiple forward passes to capture epistemic uncertainty
num_samples = 100
means = []
stds = []

for _ in range(num_samples):
    y_pred_dist = model(X_test)
    means.append(y_pred_dist.mean().numpy())
    stds.append(y_pred_dist.stddev().numpy())  # Aleatoric uncertainty

means = np.array(means)
stds = np.array(stds)

# Calculate different uncertainty types
pred_mean = np.mean(means, axis=0)  # Predictive mean
aleatoric_uncertainty = np.mean(stds**2, axis=0)  # Average aleatoric uncertainty
epistemic_uncertainty = np.var(means, axis=0)  # Variance in means (epistemic)
total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

# Create plots
plt.figure(figsize=(15, 10))

# Plot 1: Data and prediction
plt.subplot(2, 2, 1)
plt.scatter(X_train, y_train, alpha=0.4, label='Training data')
plt.scatter(X_full, y_full, alpha=0.1, color='gray', label='Full data distribution')
plt.plot(X_test, pred_mean, 'r-', linewidth=2, label='Predictive mean')
plt.fill_between(X_test.reshape(-1), 
                 pred_mean.reshape(-1) - 2 * np.sqrt(total_uncertainty).reshape(-1),
                 pred_mean.reshape(-1) + 2 * np.sqrt(total_uncertainty).reshape(-1),
                 color='red', alpha=0.2, label='Total uncertainty (±2σ)')
plt.title('Bayesian NN Prediction with Uncertainty')
plt.legend()
plt.grid(True)

# Plot 2: Aleatoric uncertainty
plt.subplot(2, 2, 2)
plt.plot(X_test, np.sqrt(aleatoric_uncertainty), 'b-', label='Aleatoric uncertainty (std)')
plt.title('Aleatoric Uncertainty (Data Noise)')
plt.xlabel('X')
plt.ylabel('Standard Deviation')
plt.grid(True)
plt.legend()

# Plot 3: Epistemic uncertainty
plt.subplot(2, 2, 3)
plt.plot(X_test, np.sqrt(epistemic_uncertainty), 'g-', label='Epistemic uncertainty (std)')
plt.title('Epistemic Uncertainty (Model Uncertainty)')
plt.xlabel('X')
plt.ylabel('Standard Deviation')
plt.axvspan(-0.5, 0.5, color='gray', alpha=0.2, label='Data gap')
plt.grid(True)
plt.legend()

# Plot 4: Comparison of uncertainty types
plt.subplot(2, 2, 4)
plt.plot(X_test, np.sqrt(aleatoric_uncertainty), 'b-', label='Aleatoric')
plt.plot(X_test, np.sqrt(epistemic_uncertainty), 'g-', label='Epistemic')
plt.plot(X_test, np.sqrt(total_uncertainty), 'r-', label='Total')
plt.title('Comparison of Uncertainty Types')
plt.xlabel('X')
plt.ylabel('Standard Deviation')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Calculate calibration curve (reliability diagram)
def calculate_calibration_curve(model, X, y, num_bins=10):
    # Get predictions
    y_pred_dist = model(X)
    mean_preds = y_pred_dist.mean().numpy().flatten()
    std_preds = y_pred_dist.stddev().numpy().flatten()
    
    # Calculate standardized residuals
    z_scores = np.abs((y.flatten() - mean_preds) / std_preds)
    
    # For a well-calibrated model, about 68% of residuals should be within 1 std dev,
    # and about 95% should be within 2 std devs
    
    # Calculate fraction of points within different confidence levels
    confidence_levels = np.linspace(0.1, 3.0, num_bins)
    observed_freq = []
    expected_freq = []
    
    for c in confidence_levels:
        # Calculate expected frequency from normal CDF
        expected = 2 * (stats.norm.cdf(c) - 0.5)  # Convert to percentage
        expected_freq.append(expected)
        
        # Calculate observed frequency
        observed = np.mean(z_scores <= c)
        observed_freq.append(observed)
    
    return confidence_levels, observed_freq, expected_freq

try:
    from scipy import stats
    
    # Plot calibration curve
    conf_levels, obs_freq, exp_freq = calculate_calibration_curve(model, X_test, 
                                                                 pred_mean.reshape(-1, 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(exp_freq, exp_freq, 'r--', label='Ideal calibration')
    plt.plot(exp_freq, obs_freq, 'bo-', label='Model calibration')
    plt.title('Uncertainty Calibration Curve')
    plt.xlabel('Expected Confidence Level')
    plt.ylabel('Observed Frequency')
    plt.grid(True)
    plt.legend()
    plt.show()
except ImportError:
    print("SciPy not found. Skipping calibration curve.")

print("Key observations:")
print("1. Aleatoric uncertainty (data noise) varies across the input space")
print("2. Epistemic uncertainty (model uncertainty) is highest in regions with no training data")
print("3. The model correctly identifies different sources of uncertainty")