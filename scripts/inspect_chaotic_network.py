import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list

from external.chaotic_network_John import simulate_chaotic_network, resample_fixed_times
from chaotic_network import initialize_chaotic_network, chaotic_step


def sort_neurons_by_similarity(X, method='average'):
    # X shape (T, N)
    Xc = X - X.mean(axis=0)
    C = np.corrcoef(Xc.T)
    D = 1 - C
    order = leaves_list(linkage(squareform(D), method=method))
    return X[:, order], order

def plot_chaotic_network_activity(x, title, dt, pos, nrows=4):
    steps, N = x.shape
    T = steps * dt
    ax = plt.subplot(nrows, 1, pos)
    im = ax.imshow(x.T, aspect='auto', interpolation='none',
                   extent=[0, T, 0, N], origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_ylabel("Neuron index")
    ax.set_xlabel("Time (s)")
    return ax

def plot_activities(xs, hs, qs, Ws, dt, steps):   
    fig, ax = plt.subplots(4, 1, figsize=(8, 10), layout='tight')
    plt.set_cmap('RdBu')

    plot_chaotic_network_activity(xs, "Neuron activities", dt, 1)
    plot_chaotic_network_activity(hs, "Membrane potentials", dt, 2)
    plot_chaotic_network_activity(qs,
                                  "Running average of second moment of activity", dt, 3)
    W_plot = Ws.reshape((steps, -1))
    n_weights_plot = 300
    W_plot = W_plot[:, :np.min((n_weights_plot, N*N))]
    plot_chaotic_network_activity(W_plot,
                                  f"Recurrent weights (first {n_weights_plot} weights)", dt, 4)
    ax[3].set_ylabel("Weight index")
    plt.show()

def plot_top_PC_eigenvalues(X, dt):
    w = 100  # window length in samples
    step = 50

    times = []
    V = np.zeros((X.shape[1], (X.shape[0]-w)//step + 1))
    i = 0
    for t in range(0, X.shape[0]-w+1, step):
        Xw = X[t:t+w]
        Xw = Xw - Xw.mean(axis=0)
        U,S,Vt = np.linalg.svd(Xw, full_matrices=False)
        var = S**2
        var = var / var.sum()
        var_sorted = np.sort(var)[::-1]
        V[:var_sorted.size, i] = var_sorted
        times.append(t + w/2)
        i += 1

    fig,ax = plt.subplots()
    im = ax.imshow(V[:7], aspect='auto', origin='lower', interpolation='none', extent=[0, (V.shape[1]-1)*dt, 0, 7])
    ax.set_xlabel('Time (s) (window center)')
    ax.set_ylabel('PC index (sorted)')
    plt.colorbar(im, label='Explained variance')
    plt.xticks(
        np.linspace(0, (V.shape[1]-1)*dt, 5), 
        np.linspace(times[0]*dt, times[-1]*dt, 5, dtype=int)
        )
    plt.show()

def plot_2D_trajectories(X, dt):
    Xc = X - X.mean(axis=0)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    PC = Xc.dot(Vt.T)[:, :2]                 # (T,2) projected trajectory
    segments = np.stack([PC[:-1], PC[1:]], axis=1)
    lc = LineCollection(segments, cmap='rainbow', norm=plt.Normalize(0, (PC.shape[0]-1)*dt))
    lc.set_array(np.arange(PC.shape[0]-1)*dt)
    lc.set_linewidth(2)

    fig,ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.colorbar(lc, ax=ax, label='Time (s)')
    plt.show()

def main(version='my'):
    x, h, W, q = initialize_chaotic_network(N, g)

    if version == 'my':
      steps = int(Tm / dt)
      xs, hs, qs = [np.zeros((steps, N)) for _ in range(3)]
      Ws = np.zeros((steps, N, N))
      for i in range(steps):
          x, h, W, q = chaotic_step(x, h, W, q, g, alpha, tau_h, tau_q, dt)
          xs[i], hs[i], Ws[i], qs[i] = x, h, W, q

    elif version == 'John':
      t_cont, y_cont = simulate_chaotic_network(h, W, q)
      t, y = resample_fixed_times(t_cont, y_cont, Tm)

      Ws = np.reshape(y[:N*N], (N, N))  # weight matrix
      hs = y[N*N:N*N+N]  # membrane potentials (leaky integrator of inputs)
      qs = y[N*N+N:N*N+2*N]  # running average of second moment of activity

      steps = t.shape[0]
      xs = np.tanh(g * hs)  # Neuron's response

    xs, order = sort_neurons_by_similarity(xs)
    hs = hs[:, order]
    qs = qs[:, order]
    Ws = Ws[:, order][:, :, order]

    plot_activities(xs, hs, qs, Ws, dt, steps)
    plot_top_PC_eigenvalues(xs, dt)
    plot_2D_trajectories(xs, dt)

if __name__ == "__main__":
    N = 300
    g = 2.0  # Steepness of tanh activation function (how much all-or-none the response is)
    alpha = 0.5  # Learning rate
    tau_h = 1.0  # Time constant of membrane potential dynamics
    tau_q = 30.0  # Time constant of running average of second moment of activity
    dt = 0.1  # Time step for simulation
    Tm = 1000.0  # Total simulation time

    np.random.seed(25)

    main(version='my')   # Use 'my' for own implementation, 'John' for external implementation