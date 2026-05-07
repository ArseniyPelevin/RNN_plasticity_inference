''' Chaotic network with variable balance learning rule
    by John Briguglio
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

N = 300
g = 2.0  # Steepness of tanh activation function (how much all-or-none the response is)
alpha = 0.5  # Learning rate
tau_h = 1.0  # Time constant for membrane potential
tau_q = 30.0  # Time constant for running average of second moment of activity
Tm = 300.0  # Total simulation time

def dydt_varbal(t, y, g, alpha, tau_h, tau_q):
    '''
    d/dt(h_i) = -h_i + sum_j W_{ij} tanh(g h_j)
    d/dt(q_i) = (x_i^2 - q_i) / tau_q
    d/dt(W_ij) = -alpha * (x_i^2 - q_i + x_j^2 - q_j) * W_ij
    '''
    # len(y) = N^2 + 2N; sqrt(len(y)+1) = N + 1
    N = round(np.sqrt(len(y) + 1) - 1)

    # Unpack variables from y
    W = np.reshape(y[:N*N], (N, N))  # weight matrix
    h = y[N*N:N*N+N]  # membrane potentials (leaky integrator of inputs)
    q = y[N*N+N:N*N+2*N]  # running average of second moment of activity

    x = np.tanh(g * h)  # Neuron's response
    xi2 = x**2
    # Matrix of sums of deviations of pre- and post-synaptic neurons from their averages
    M = (xi2[:, None] - q[:, None]) + (xi2[None, :] - q[None, :])
    dW = -alpha * M * W
    np.fill_diagonal(dW, 0)  # no autapses
    dh = (-h + (W @ x) / np.sqrt(N)) / tau_h
    dq = (xi2 - q) / tau_q
    return np.concatenate((dW.ravel(), dh, dq))

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

def simulate_chaotic_network(h0, W0, q0):
    ''' Simulate chaotic network with variable balance learning rule '''
    y0 = np.concatenate((W0.ravel(), h0, q0))
    sol = solve_ivp(lambda t, y: dydt_varbal(t, y, g, alpha, tau_h, tau_q), [0, Tm], y0)
    t = sol.t
    y = sol.y.T  # shape (nt, nvars)
    return t, y

def resample_fixed_times(t, y, Tm, dt=0.1):
    y_interp = interp1d(t, y, axis=0, kind='linear', fill_value='extrapolate')
    t_fixed = np.arange(0, Tm + dt, dt)
    Y = y_interp(t_fixed)  # Original trajectory sampled at fixed times
    return t_fixed, Y

def compare_trajectories(t, y, td, yd):
    # Compare trajectories to estimate Lyapunov exponent
    t_compare, Y = resample_fixed_times(t, y, Tm)  # Original trajectory sampled at fixed times
    _, YD = resample_fixed_times(td, yd, Tm)  # Perturbed trajectory sampled at fixed times
    Delta = np.linalg.norm(Y - YD, axis=1)  # Euclidean distance between trajectories
    cutoff = 1e6
    valid = Delta < Delta[0] * cutoff
    fit_param = np.polyfit(t_compare[valid], np.log(Delta[valid]), 1)
    return t_compare, Delta, valid, fit_param

def plot_x(t, h, g, N):  # (membrane potentials)
    plt.figure()
    plt.imshow(np.tanh(g * h).T, extent=[t[0], t[-1], 1, N], aspect='auto', origin='lower',
            vmin=-1, vmax=1, interpolation='none')
    plt.xlabel('time')
    plt.ylabel('N')
    plt.title('x = tanh(g*h)')
    plt.colorbar()
    plt.set_cmap('RdBu')

def plot_q(t, q, N):  # (running average of the second moment of activity)
    plt.figure()
    plt.imshow(q.T, extent=[t[0], t[-1], 1, N], aspect='auto', origin='lower',
            vmin=-1, vmax=1, interpolation='none')
    plt.xlabel('time')
    plt.ylabel('N')
    plt.title('q')
    plt.colorbar()
    plt.set_cmap('RdBu')

def plot_W(t, W_plot):  # (weights)
    plt.figure()
    plt.imshow(W_plot.T, extent=[t[0], t[-1], 1, W_plot.shape[1]],
            aspect='auto', origin='lower', interpolation='none')
    plt.xlabel('time')
    plt.ylabel('idx')
    plt.title('W')
    plt.colorbar()
    plt.set_cmap('RdBu')
    plt.clim(-1, 1)

def plot_Delta(t_compare, Delta, valid, fit_param):
    # Distance and fitted Lyapunov exponent
    plt.figure()
    plt.semilogy(t_compare, Delta)
    plt.semilogy(t_compare[valid],
                np.exp(fit_param[0] * t_compare[valid] + fit_param[1]), 'r--')
    plt.ylabel('Delta')
    plt.xlabel('time')
    ax = plt.gca()
    ax.tick_params(direction='out')
    ax.set_yscale('log')
    plt.box(False)
    plt.legend(['simulation', f'λ={fit_param[0]:.6f}'])
    plt.show()

def main():
    _x0, h0, W0, q0 = initialize_chaotic_network(N, g)

    t, y = simulate_chaotic_network(h0, W0, q0)

    # Define small perturbations
    sigh = 1e-6
    sigW = 0.0
    delta_h = np.random.normal(0, sigh**2, h0.shape)
    delta_W = np.random.normal(0, sigW**2, W0.shape)

    td, yd = simulate_chaotic_network(h0 + delta_h, W0 + delta_W, q0)

    t_compare, Delta, valid, fit_param = compare_trajectories(t, y, td, yd)
    
    # extract variables for plotting
    start = N * N
    h = y[:, start:start + N]
    q = y[:, start + N:start + 2 * N]
    W_plot = y[:, :min(1000, N * N)]

    plot_x(t, h, g, N)
    plot_q(t, q, N)
    plot_W(t, W_plot)
    plot_Delta(t_compare, Delta, valid, fit_param)

if __name__ == "__main__":
    main()