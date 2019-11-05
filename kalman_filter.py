import numpy as np

def transition_func(trans_mat, x_mean_prev_post, force_mat, u_prev):
    return trans_mat@x_mean_prev_post + force_mat@u_prev

def observation_func(obs_mat, x_mean_curr_prior):
    return obs_mat@x_mean_curr_prior

def get_kalman_gain(x_cov_curr_prior, obs_mat, obs_cov):
    return x_cov_curr_prior@(obs_mat.T)@np.linalg.inv(obs_mat@x_cov_curr_prior@(obs_mat.T) + obs_cov)

def kalman_filter_iter():
    process_dim = 2
    obs_dim = 2
    force_dim = 2

    x_mean_prev_post = np.zeros((process_dim, 1))
    x_cov_prev_post = np.zeros((process_dim, process_dim))

    obs = np.zeros((2, obs_dim))
    obs_mats = np.zeros((2, process_dim, obs_dim))
    obs_covs = np.zeros((2, obs_dim, obs_dim))

    trans_mats = np.zeros((2, process_dim, process_dim))
    trans_covs = np.zeros((2, process_dim, process_dim))

    force_mats = np.zeros((2, process_dim, force_dim))
    forces = np.zeros((2, force_dim))

    num_iters = 2

    for i in range(0, num_iters):

        x_mean_curr_prior = transition_func(
            trans_mats[i,:,:], x_mean_prev_post, force_mats[i,:,:], forces[i,:])

        x_cov_curr_prior = trans_mats[i,:,:]@x_cov_prev_post@(trans_mats[i,:,:].T) + trans_covs[i,:,:]

        kalman_gain = get_kalman_gain(
            x_cov_curr_prior, obs_mats[i,:,:], obs_covs[i,:,:])
        
        x_mean_curr_post = x_mean_curr_prior + kalman_gain@(obs[i, :] - observation_func(obs_mats[i,:,:], x_mean_curr_prior))

        x_cov_curr_pos = (np.identity(process_dim) - kalman_gain@obs_mats[i,:,:])@x_cov_curr_prior     

        x_mean_prev_post = x_mean_curr_post
        x_cov_prev_post - x_cov_curr_pos 


if __name__ == '__main__':
    kalman_filter_iter()