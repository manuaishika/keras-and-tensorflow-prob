import numpy as np

def generate_data(n=100):
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    noise = np.random.normal(0, 0.5, size=X.shape)
    Y = 2.0 * X + 1.0 + noise
    return X.astype('float32'), Y.astype('float32')
