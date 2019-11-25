import numpy as np

N = 40 # num dimensions
F = 8 # forcing

def lorenz_grad(x):
    global N
    nabla = np.zeros(N)
    nabla[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
    nabla[1] = (x[2] - x[N-1]) * x[0] - x[1]
    nabla[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
    nabla[2:N-1] = (x[3:N] - x[0:N-3]) * x[1:N-2] - x[2:N-1]
    return nabla + F

def lorenz_rk4_transition(x, prev_j, step=1e-5, mode=None):
    global N
    N = len(x)
    k = np.zeros((N, 4))
        
    k[:, 0] = step * lorenz_grad(x)
    k[:, 1] = step * lorenz_grad(x + k[:, 0] / 2)
    k[:, 2] = step * lorenz_grad(x + k[:, 1] / 2)   
    k[:, 3] = step * lorenz_grad(x + k[:, 2])

    x_new = x + (k[:, 0] + 2 * k[:, 1] + 2 * k[:, 2] + k[:, 3]) / 6

    if mode == 'jac':
        k_jacobians = np.zeros((N, N, 4))
        k_jacobians[:, :, 0] = step * lorenz_jacobian(x)
        k_jacobians[:, :, 1] = step * (prev_j + k_jacobians[:, :, 0] / 2) \
            * lorenz_jacobian(x + k[:, 0] / 2)
        k_jacobians[:, :, 2] = step * (prev_j + k_jacobians[:, :, 1] / 2) \
            * lorenz_jacobian(x + k[:, 1] / 2)
        k_jacobians[:, :, 3] = step * (prev_j + k_jacobians[:, :, 2]) \
            * lorenz_jacobian(x + k[:, 2])
        trans_mat = prev_j + (k_jacobians[:, :, 0] +  k_jacobians[:, :, 1] + \
            k_jacobians[:, :, 2] + k_jacobians[:, :, 3]) / 6

        return x_new, trans_mat

    return x_new
    

def lorenz_jacobian(x):
    global N
    J = np.identity(N) * -1

    J[0, 1] = x[N-1]
    J[0, N-2] = -x[N-1]
    J[0, N-1] = x[1] - x[N-2]

    J[1, 0] = x[2] - x[N-1]
    J[1, 2] = x[1]
    J[1, N-1] = x[1]

    J[N-1, 0] = x[N-2]
    J[N-1, N-3] = -x[N-2]
    J[N-1, N-2] = x[0] - x[N-3]

    for i in range(2, N-1):
        J[i, i-2] = -x[i-1]
        J[i, i-1] = x[i+1] - x[i-2]
        J[i, i+1] = x[i-1]

    return J

def init_jacobian(x):
    global N
    N = len(x)
    J = np.zeros((N, N))
    diff = 1e-6
    ts = 1e-3
    for i in range(N):
        J[:, i] = approx_pderiv_col(x, i, diff, ts)
    
    return J

def approx_pderiv_col(x, i, diff, ts):
    a = x.copy()
    a[i] -= diff
    b = x.copy()
    b[i] += diff
    return (lorenz_rk4_transition(a, 0, step=ts)[0] - lorenz_rk4_transition(b, 0, step=ts)[0]) / 2 / diff
