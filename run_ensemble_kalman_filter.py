import numpy as np
import matplotlib.pyplot as plt
from kalman_utils import ensemble_kf_iter
from lorenz_96 import lorenz_rk4_transition

def main():
    n_iters = 200
    n_ensembles = 100

    process_dim = 40
    x0 = np.ones(process_dim, dtype=np.float64) / 10
    x0[2] += 0.01

    obs_dim = 40

    true_x = np.zeros((n_iters + 1, process_dim))
    true_x[0, :] = x0
    trans_noise = np.random.normal(0, 0.5, (n_iters, process_dim))

    obs = np.zeros((n_iters, obs_dim))
    obs_noise = np.random.normal(0, 2, (n_iters, obs_dim))

    for i in range(1, n_iters+1):
        true_x[i, :] = lorenz_rk4_transition(
            np.array(true_x[i-1, :]), 0, step=0.02, mode=None)
        true_x[i, :] += trans_noise[i-1, :]
        obs[i-1, :] = true_x[i, :] + obs_noise[i-1, :]

    input_data = {
        'process_dim': process_dim,
        'x_mean_prev_post': x0,
        'x_cov_prev_post': np.identity(process_dim) * 0.25,
        'obs': obs,
        'obs_mats': np.tile(np.identity(obs_dim), (n_iters, 1, 1)),
        'obs_covs': np.tile(4 * np.identity(obs_dim), (n_iters, 1, 1)),
        'transition_f': lambda x: lorenz_rk4_transition(x, 0, step=0.02, mode=None),
        'trans_covs': np.tile(0.25 * np.identity(process_dim), (n_iters, 1, 1)),
    }

    res = ensemble_kf_iter(n_iters, n_ensembles, **input_data)[0]

    x_scale = list(range(1, n_iters+1))
    kf_curve, = plt.plot(x_scale, res[:, 1], 'b-', linewidth=0.5)
    true_curve, = plt.plot([0] + x_scale, true_x[:, 1], 'g-', linewidth=0.5)
    obs_curve, = plt.plot(x_scale, obs[:, 1], 'k--', linewidth=0.1)


    plt.legend((kf_curve, true_curve, obs_curve),
        ('Kalman Filter', 'True Process', 'Observations'),
        loc='best')

    plt.show()

if __name__ == '__main__':
    main()