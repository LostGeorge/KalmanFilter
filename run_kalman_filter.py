import numpy as np
import matplotlib.pyplot as plt
import random
from kalman_utils import kalman_filter_iter

def main():

    n_iters = 20

    x0 = np.array([1.0])
    process_dim = 1

    true_x = np.zeros(n_iters + 1)
    true_x[0] = x0[0]
    trans_noise = np.random.normal(0, 0.2, n_iters)
    trans_mats = np.tile(np.zeros((1,1)), (n_iters, 1, 1))

    obs = np.zeros((n_iters, 3))
    obs_noise = np.random.normal(0, 0.4, (n_iters, 3))

    for i in range(1, n_iters+1):
        trans_mats[i-1, :, :] = np.array([[random.random() + 0.5]])
        true_x[i] =  trans_mats[i-1, :, :].item() * true_x[i-1] + trans_noise[i-1]
        obs[i-1,:] = true_x[i] + obs_noise[i-1, :]

    input_data = {
        'process_dim': process_dim,
        'x_mean_prev_post': x0,
        'x_cov_prev_post': np.array([[0.05]]),
        'obs': obs,
        'obs_mats': np.tile(np.ones((3, 1)), (n_iters, 1, 1)),
        'obs_covs': np.tile(0.16 * np.identity(3), (n_iters, 1, 1)),
        'trans_mats': trans_mats,
        'trans_covs': np.tile(0.04, (n_iters, 1, 1)),
    }

    res = kalman_filter_iter(n_iters, **input_data)

    mean_obs = np.mean(obs, axis=1)
    print(f'MSE for observation mean: \t {np.sum((mean_obs - np.array(true_x[1:]))**2):.3f}')
    print(f'MSE for Kalman Filter mean: \t {np.sum((res[:, 0] - np.array(true_x[1:]))**2):.3f}')

    x_scale = list(range(1, n_iters+1))
    plt.plot(x_scale, res[:, 0], 'b-')
    plt.plot([0] + x_scale, true_x, 'g-')
    plt.plot(x_scale, obs[:, 0], 'k--', linewidth=0.5)
    plt.plot(x_scale, obs[:, 1], 'k--', linewidth=0.5)
    plt.plot(x_scale, obs[:, 2], 'k--', linewidth=0.5)
    plt.plot(x_scale, mean_obs, 'r-')
    plt.show()
    """ for i, re in enumerate(res):
        print('%.3f' % true_x[i+1], end='\t')
        print('%.3f' % obs[i,:], end='\t')
        print(re, end='\n') """

if __name__ == '__main__':
    main()