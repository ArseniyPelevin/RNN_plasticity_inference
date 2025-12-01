''' Chaotic network with variable balance learning rule
    by John Briguglio
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def dydt_varbal(t, y, g, alpha, tau_q):

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
    M = (xi2[:, None] - q[:, None]) + (xi2[None, :] - q[None, :])  # Matrix of sums of deviations of pre- and post-synaptic neurons from their averages
    dW = -alpha * M * W
    np.fill_diagonal(dW, 0)  # no autapses
    dh = -h + (W @ x) / np.sqrt(N)
    dq = (xi2 - q) / tau_q
    return np.concatenate((dW.ravel(), dh, dq))

N = 300
g = 2.0  # Steepness of tanh activation function (how much all-or-none the response is)
alpha = 0.5  # Learning rate
tau_q = 30.0  # Time constant for running average of second moment of activity
Tm = 100.0  # Total simulation time

h0 = np.random.normal(0, 1, N)
W0 = np.random.normal(0, 1, (N, N))
# clip initial weights to [-1, 1]
W0 = np.clip(W0, -1, 1)
np.fill_diagonal(W0, 0)  # no autapses
q0 = 0.5 * np.ones(N)

y0 = np.concatenate((W0.ravel(), h0, q0))
sol = solve_ivp(lambda t, y: dydt_varbal(t, y, g, alpha, tau_q), [0, Tm], y0)
t = sol.t
y = sol.y.T  # shape (nt, nvars)

# Measure Lyapunov exponent
# pick size of delta vector
sigh = 1e-6
sigW = 0.0
delta_h = np.random.normal(0, sigh**2, h0.shape)
delta_W = np.random.normal(0, sigW**2, W0.shape)
hd = h0 + delta_h
Wd = W0 + delta_W
qd = q0.copy()
yd0 = np.concatenate((Wd.ravel(), hd, qd))
sold = solve_ivp(lambda t, y: dydt_varbal(t, y, g, alpha, tau_q), [0, Tm], yd0)
td = sold.t
yd = sold.y.T

# Compare at the same times
y_interp = interp1d(t, y, axis=0, kind='linear', fill_value='extrapolate')
yd_interp = interp1d(td, yd, axis=0, kind='linear', fill_value='extrapolate')
t_compare = np.arange(0, Tm + 0.1, 0.1)
Y = y_interp(t_compare)
YD = yd_interp(t_compare)
Delta = np.linalg.norm(Y - YD, axis=1)
cutoff = 1e6
valid = Delta < Delta[0] * cutoff
fit_param = np.polyfit(t_compare[valid], np.log(Delta[valid]), 1)

# extract variables for plotting
start = N * N
h = y[:, start:start + N]
q = y[:, start + N:start + 2 * N]
W_plot = y[:, :min(1000, N * N)]

# Plot h (membrane potentials)
plt.figure()
plt.imshow(np.tanh(g * h).T, extent=[t[0], t[-1], 1, N], aspect='auto', origin='lower',
           vmin=-1, vmax=1, interpolation='none')
plt.xlabel('time')
plt.ylabel('N')
plt.title('h')
plt.colorbar()
plt.set_cmap('RdBu')

# Plot q (running average of the second moment of activity)
plt.figure()
plt.imshow(q.T, extent=[t[0], t[-1], 1, N], aspect='auto', origin='lower',
           vmin=-1, vmax=1, interpolation='none')
plt.xlabel('time')
plt.ylabel('N')
plt.title('q')
plt.colorbar()
plt.set_cmap('RdBu')

# Plot W (synaptic weights)
plt.figure()
plt.imshow(W_plot.T, extent=[t[0], t[-1], 1, W_plot.shape[1]],
           aspect='auto', origin='lower', interpolation='none')
plt.xlabel('time')
plt.ylabel('idx')
plt.title('W')
plt.colorbar()
plt.set_cmap('RdBu')
plt.clim(-1, 1)

# Plot Delta and fitted Lyapunov exponent
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
