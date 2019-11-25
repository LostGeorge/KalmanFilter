import numpy as np
import matplotlib.pyplot as plt
from kalman_utils import kalman_filter_iter
from lorenz_96 import lorenz_rk4_transition, init_jacobian

def main():
    # TODO: Initial Jacobian and covariance, figure out some values for other covs
    n_iters = 200

    process_dim = 40
    x0 = np.ones(process_dim, dtype=np.float64) / 10
    x0[2] += 0.01

    obs_dim = 40

    true_x = np.zeros((n_iters + 1, process_dim))
    true_x[0, :] = x0
    trans_noise = np.random.normal(0, 0.2, (n_iters, process_dim))
    trans_mats = np.tile(np.zeros((process_dim, process_dim)), (n_iters + 1, 1, 1))
    # assign x0_jacobian to trans_mats[0, :, :]
    trans_mats[0, :, :] = init_jacobian(x0)
    #print(trans_mats[0, 0:5, 0:5])
    #trans_mats[0, :, :] = np.identity(process_dim) * 2

    obs = np.zeros((n_iters, obs_dim))
    obs_noise = np.random.normal(0, 2, (n_iters, obs_dim))


    for i in range(1, n_iters+1):
        true_x[i, :], trans_mats[i, :, :] = lorenz_rk4_transition(
            np.array(true_x[i-1, :]), trans_mats[i-1, :, :], step=0.05, mode='jac')
        true_x[i, :] += trans_noise[i-1, :]
        obs[i-1, :] = true_x[i, :] + obs_noise[i-1, :]

    input_data = {
        'process_dim': process_dim,
        'x_mean_prev_post': x0,
        'x_cov_prev_post': np.identity(process_dim) * 0.04,
        'obs': obs,
        'obs_mats': np.tile(np.identity(obs_dim), (n_iters, 1, 1)),
        'obs_covs': np.tile(4 * np.identity(obs_dim), (n_iters, 1, 1)),
        'trans_mats': trans_mats[1:, :, :],
        'trans_covs': np.tile(0.04 * np.identity(process_dim), (n_iters, 1, 1)),
    }

    res = kalman_filter_iter(n_iters, 'matrix', **input_data)[0]

    #mean_obs = np.mean(obs, axis=1)
    #print(f'MSE for observation mean: \t {np.sum((mean_obs - np.array(true_x[1:]))**2):.3f}')
    #print(f'MSE for Kalman Filter: \t {np.sum((res[:, 0] - np.array(true_x[1:]))**2):.3f}')

    x_scale = list(range(1, n_iters+1))
    kf_curve, = plt.plot(x_scale, res[:, 1], 'b-', linewidth=0.5)
    true_curve, = plt.plot([0] + x_scale, true_x[:, 1], 'g-', linewidth=0.5)
    obs_curve, = plt.plot(x_scale, obs[:, 1], 'k--', linewidth=0.1)
   # mean_obs_curve, = plt.plot(x_scale, mean_obs, 'r-', linewidth=0.5)

    plt.legend((kf_curve, true_curve, obs_curve),
        ('Kalman Filter', 'True Process', 'Observations'),
        loc='best')

    plt.show()

if __name__ == '__main__':
    main()