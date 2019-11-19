import numpy as np

N = 12 # num dimensions
F = 8 # forcing

def lorenz_grad(x):
    global N
    nabla = np.zeros(N)
    nabla[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
    nabla[1] = (x[2] - x[N-1]) * x[0] - x[1]
    nabla[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
    return nabla + F

def lorenz_transition_matrix(x, step=1e-5):
    global N
    N = len(x)
    k = np.zeros((N, 4))
    k[:, 0] = step * lorenz_grad(x)
    k[:, 1] = step * lorenz_grad(x + k[:, 0] / 2)
    k[:, 2] = step * lorenz_grad(x + k[:, 1] / 2)
    k[:, 3] = step * lorenz_grad(x + k[:, 2])
    x_new = x + (k[:, 0] + 2 * k[:, 1] + 2 * k[:, 2] + k[:, 3]) / 6
    scalar_mat = np.zeros((N, N))
    for i in range(N):
        scalar_mat[i, i] = x_new[i] / x[i]
    return scalar_mat
