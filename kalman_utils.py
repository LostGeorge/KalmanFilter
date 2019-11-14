import numpy as np

def transition_func(trans_mat, x_mean_prev_post, mode, force_mat=np.zeros((1,1)), u_prev=np.zeros(1)):
    if mode == 'matrix':
        return trans_mat@x_mean_prev_post + force_mat@u_prev
    else: # mode is the transition function
        return mode(x_mean_prev_post)@x_mean_prev_post + force_mat@u_prev
    
def observation_func(obs_mat, x_mean_curr_prior):
    return obs_mat@x_mean_curr_prior

def get_kalman_gain(x_cov_curr_prior, obs_mat, obs_cov):
    return x_cov_curr_prior@(obs_mat.T)@np.linalg.inv(obs_mat@x_cov_curr_prior@(obs_mat.T) + obs_cov)

def kalman_filter_iter(num_iters, mode, **data_dict):
    
    results = np.zeros((num_iters, 2))

    process_dim = data_dict['process_dim']
    
    x_mean_prev_post = data_dict['x_mean_prev_post']
    x_cov_prev_post = data_dict['x_cov_prev_post']
    
    trans_mats = data_dict['trans_mats']
    trans_covs = data_dict['trans_covs']
    
    obs = data_dict['obs']
    obs_mats = data_dict['obs_mats']
    obs_covs = data_dict['obs_covs']

    forces =  np.tile(np.zeros(1), (num_iters, 1))
    force_mats = np.tile(np.zeros((1,1)), (num_iters, 1, 1))
    if 'forces' in data_dict:
        forces =  data_dict['forces']
        force_mats = data_dict['force_mats']

    for i in range(0, num_iters):
        x_mean_curr_prior = transition_func(
            trans_mats[i,:,:], x_mean_prev_post, mode, force_mats[i,:,:], forces[i,:])

        x_cov_curr_prior = trans_mats[i,:,:]@x_cov_prev_post@(trans_mats[i,:,:].T) + trans_covs[i,:,:]

        kalman_gain = get_kalman_gain(
            x_cov_curr_prior, obs_mats[i,:,:], obs_covs[i,:,:])
        
        x_mean_curr_post = x_mean_curr_prior + kalman_gain@(obs[i, :] - observation_func(obs_mats[i,:,:], x_mean_curr_prior))
        
        x_cov_curr_pos = (np.identity(process_dim) - kalman_gain@obs_mats[i,:,:])@x_cov_curr_prior     

        x_mean_prev_post = x_mean_curr_post
        x_cov_prev_post = x_cov_curr_pos 

        results[i, :] = [x_mean_prev_post, x_cov_prev_post]

    return results