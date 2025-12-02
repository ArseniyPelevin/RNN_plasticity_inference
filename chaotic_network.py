import numpy as np


def initialize_chaotic_network(N, g=2.0):
    ''' Initialize chaotic network with variable balance learning rule '''
    h0 = np.random.normal(0, 1, N)
    x0 = np.tanh(g * h0)
    W0 = np.random.normal(0, 1, (N, N))
    # clip initial weights to [-1, 1]
    W0 = np.clip(W0, -1, 1)
    np.fill_diagonal(W0, 0)  # no autapses
    q0 = 0.5 * np.ones(N)
    return x0, h0, W0, q0

def chaotic_step(x, h, W, q, g=2.0, alpha=0.5, tau_h = 1.0, tau_q=30.0, dt=0.1):
    ''' Single time step of chaotic network with variable balance learning rule '''
    N = len(x)
    beta_h = dt / tau_h  # 0.1 / 1.0 = 0.1
    beta_q = dt / tau_q  # 0.1 / 30.0 = 0.0033
    beta_W = dt * alpha  # 0.1 * 0.5 = 0.05

    # Update running average of second moment of activity
    q = (1 - beta_q) * q + beta_q * (x**2)

    # Update weights based on deviations from average activity
    M = ((x**2)[:, None] - q[:, None]) + ((x**2)[None, :] - q[None, :])
    dW = - beta_W * M * W
    np.fill_diagonal(dW, 0)  # no autapses

    W = W + dW

    # Compute network input
    inp = (W @ x) / np.sqrt(N)

    # Update membrane potentials (leaky integrator of inputs)
    h = (1 - beta_h) * h + beta_h * inp
    # Update neuron responses
    x = np.tanh(g*h)

    return x, h, W, q
