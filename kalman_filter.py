import numpy as np

def transition_func(trans_mat, x_mean_prev_post, force_mat=np.zeros((1,1)), u_prev=np.zeros(1)):
    return trans_mat@x_mean_prev_post + force_mat@u_prev

def observation_func(obs_mat, x_mean_curr_prior):
    return obs_mat@x_mean_curr_prior

def get_kalman_gain(x_cov_curr_prior, obs_mat, obs_cov):
    return x_cov_curr_prior@(obs_mat.T)@np.linalg.inv(obs_mat@x_cov_curr_prior@(obs_mat.T) + obs_cov)

def kalman_filter_iter(num_iters, **data_dict):
    
    results = []

    process_dim = data_dict['process_dim']
    
    x_mean_prev_post = data_dict['x_mean_prev_post']
    x_cov_prev_post = data_dict['x_cov_prev_post']
    
    trans_mats = data_dict['trans_mats']
    trans_covs = data_dict['trans_covs']
    
    obs = data_dict['obs']
    obs_mats = data_dict['obs_mats']
    obs_covs = data_dict['obs_covs']

    if 'forces' in data_dict:
        forces =  data_dict['forces']
        force_mats = data_dict['force_mats']

    for i in range(0, num_iters):
        #print(x_mean_prev_post)
        x_mean_curr_prior = transition_func(
            trans_mats[i,:,:], x_mean_prev_post)

        x_cov_curr_prior = trans_mats[i,:,:]@x_cov_prev_post@(trans_mats[i,:,:].T) + trans_covs[i,:,:]

        kalman_gain = get_kalman_gain(
            x_cov_curr_prior, obs_mats[i,:,:], obs_covs[i,:,:])
        
        x_mean_curr_post = x_mean_curr_prior + kalman_gain@(obs[i, :] - observation_func(obs_mats[i,:,:], x_mean_curr_prior))
        
        x_cov_curr_pos = (np.identity(process_dim) - kalman_gain@obs_mats[i,:,:])@x_cov_curr_prior     
   
        x_mean_prev_post = x_mean_curr_post
        x_cov_prev_post - x_cov_curr_pos 
        
        results.append([x_mean_prev_post, x_cov_prev_post])

    return results

def main():

    n_iters = 30

    process_dim = 1
    obs_dim = 1
    #force_dim = 2

    input_data = {
        'process_dim': process_dim,
        'x_mean_prev_post': np.array([0.65]),
        'x_cov_prev_post': np.array([[0.05]]),
        'obs': np.random.rand(30, 1) * 2 - np.ones((30,1)),
        'obs_mats': np.array([[[1]] for i in range(10)] + [[[0.1]] for i in range(10)] + [[[1]] for i in range(10)]),
        'obs_covs': np.array([[[0.01]] for i in range(30)]),
        'trans_mats': np.array([[[0.8]] for i in range(30)]),
        'trans_covs': np.array([[[0.16]] for i in range(30)]),
        #'force_mats': np.zeros((2, process_dim, force_dim)),
        #'forces': np.zeros((2, force_dim))
    }

    res = kalman_filter_iter(n_iters, **input_data)

    print(input_data['obs'])

    for re in res:
        print(re, end='\n')

if __name__ == '__main__':
    main()