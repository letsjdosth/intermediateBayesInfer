import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

rn_generator = np.random.default_rng(seed=20220216)

def gumbel2_inverse_cdf(y, param_alpha, param_beta):
    return (-np.log(y)/param_beta)**(-1/param_alpha)

def gumbel2_random_number_generator(num_smpl, param_alpha, param_beta):
    unif_smpl = rn_generator.random(num_smpl)
    gumbel2_smpl = np.array([gumbel2_inverse_cdf(y, param_alpha, param_beta) for y in unif_smpl])
    return gumbel2_smpl

# generate sample
gumbel2_smpl_55 = gumbel2_random_number_generator(500, 5, 5)

# print(gumbel2_smpl_55)
# plt.hist(gumbel2_smpl_55, 15)
# plt.show()

# posterior
def unif_gumbel2_log_posterior(prior_param_vec, data):
    #log q (in the note)
    prior_alpha, prior_beta = prior_param_vec
    n = data.size
    negalpha_powered_data_sum = np.sum(data**(-prior_alpha))
    logdata_sum = np.sum(np.log(data))

    log_q_val = (n * (np.log(prior_alpha) + np.log(prior_beta))
        + (1-prior_alpha)*logdata_sum
        - prior_beta*negalpha_powered_data_sum)

    return log_q_val


#cantour plot of log-posterior
grid_1d = [i for i in np.arange(3, 7, 0.1)]
grid_1d_mesh_X, grid_1d_mesh_Y=np.meshgrid(grid_1d, grid_1d)
grid_2d = [(i,j) for i in np.arange(3, 7, 0.1) for j in np.arange(3, 7, 0.1)] #40*40

log_q_val_vec = []
for param in grid_2d:
    log_q_val_vec.append(unif_gumbel2_log_posterior(param, gumbel2_smpl_55))
log_q_val_vec = np.array(log_q_val_vec).reshape((40,40)).transpose()

plt.contour(grid_1d_mesh_X, grid_1d_mesh_Y,log_q_val_vec, levels=100) 
# plt.show()


#Laplace approximation

class NewtonUnconstrained:
    def __init__(self, fn_objective, fn_objective_gradient, fn_objective_hessian, fn_objective_domain_indicator = None):
        
        self.objective = fn_objective
        self.objective_gradient = fn_objective_gradient
        self.objective_hessian = fn_objective_hessian
        
        
        if fn_objective_domain_indicator is not None:
            self.objective_domain_indicator = fn_objective_domain_indicator
        else:
            self.objective_domain_indicator = self._Rn_domain_indicator
        
        self.minimizing_sequence = []
        self.decrement_sequence = []
        self.value_sequence = []
        

    def _Rn_domain_indicator(self, eval_pt):
        return True

    def _backtracking_line_search(self, eval_pt, descent_direction, 
            a_slope_flatter_ratio, b_step_shorten_ratio):

        if a_slope_flatter_ratio <= 0 or a_slope_flatter_ratio >= 0.5:
            raise ValueError("a should be 0 < a < 0.5")
        if b_step_shorten_ratio <= 0 or b_step_shorten_ratio >= 1:
            raise ValueError("b should be 0 < a < 1")

        step_size = 1

        while True:
            flatten_line_slope = self.objective_gradient(eval_pt) * a_slope_flatter_ratio * step_size
            deviation_vec = descent_direction * step_size

            objective_fn_value = self.objective(eval_pt + deviation_vec)
            flatten_line_value = self.objective(eval_pt) + sum(flatten_line_slope * deviation_vec)

            if objective_fn_value < flatten_line_value and self.objective_domain_indicator(eval_pt + deviation_vec):
                break
            else:
                step_size = step_size * b_step_shorten_ratio

        return step_size
    
    def _l2_norm(self, vec):
        return (sum(vec**2))**0.5
    
    def _descent_direction_newton_cholesky(self, eval_pt):
        hessian = self.objective_hessian(eval_pt)
        neg_gradient = self.objective_gradient(eval_pt) * (-1)
        cholesky_lowertri_of_hessian = scipy.linalg.cholesky(hessian, lower=True) # L ; H = L(L^T)
        #want: x; Hx = -g
        forward_variable = scipy.linalg.solve_triangular(cholesky_lowertri_of_hessian, neg_gradient, lower=True) # w ; Lw = -g
        newton_decrement = self._l2_norm(forward_variable)
        direction_newton_step = scipy.linalg.solve_triangular(np.transpose(cholesky_lowertri_of_hessian), forward_variable, lower=False) # x ; (L^t)x = w
        return (direction_newton_step, newton_decrement)

    # later: sparse / band hessian version

    def _descent_direction_newton_pinv(self, eval_pt):
        hessian = self.objective_hessian(eval_pt)
        neg_gradient = self.objective_gradient(eval_pt) * (-1)
        direction_newton_step = np.matmul(np.linalg.pinv(hessian), neg_gradient) #pseudo
        newton_decrement = np.matmul(np.transpose(neg_gradient), direction_newton_step)
        return (direction_newton_step, newton_decrement)

    def run_newton_with_backtracking_line_search(self, starting_pt, tolerance = 0.001, 
                                                method="cholesky", 
                                                a_slope_flatter_ratio = 0.2, b_step_shorten_ratio = 0.5):
        #method : cholesky, pinv (if hessian is singular)
        self.minimizing_sequence = [starting_pt]
        self.value_sequence = [self.objective(starting_pt)]
        self.decrement_sequence = []
        num_iter = 0
        while True:
            eval_pt = self.minimizing_sequence[-1]
            if method == "cholesky":
                descent_direction, decrement = self._descent_direction_newton_cholesky(eval_pt)
            elif method == "pinv":
                descent_direction, decrement = self._descent_direction_newton_pinv(eval_pt)
            else:
                raise ValueError("method should be ['cholesky', 'pinv']")
            self.decrement_sequence.append(decrement)

            if (decrement**2) < (tolerance*2):
                break

            descent_step_size = self._backtracking_line_search(eval_pt, descent_direction, 
                                    a_slope_flatter_ratio, b_step_shorten_ratio)
            next_point = eval_pt + descent_direction * descent_step_size
            self.minimizing_sequence.append(next_point)
            self.value_sequence.append(self.objective(next_point))
            num_iter += 1

        print("iteration: ", num_iter)
    
    def get_minimizing_sequence(self):
        return self.minimizing_sequence
    
    def get_minimizing_function_value_sequence(self):
        return self.value_sequence

    def get_decrement_sequence(self):
        return self.decrement_sequence

    def get_arg_min(self):
        return self.minimizing_sequence[-1]

    def get_min(self):
        return self.objective(self.minimizing_sequence[-1])

def unif_gumbel2_log_posterior_neg(prior_param_vec, data):
    return unif_gumbel2_log_posterior(prior_param_vec, data)*(-1)

def unif_gumbel2_log_posterior_neg_gradient(prior_param_vec, data):
    #dlog q/da, logq/db (in the note)
    prior_alpha, prior_beta = prior_param_vec
    n = data.size
    negalpha_powered_data_times_log_data_sum = np.sum(data**(-prior_alpha) * np.log(data))
    negalpha_powered_data_sum = np.sum(data**(-prior_alpha))
    logdata_sum = np.sum(np.log(data))

    dlogq_da = n/prior_alpha - logdata_sum + prior_beta * negalpha_powered_data_times_log_data_sum
    dlogq_db = n/prior_beta - negalpha_powered_data_sum

    return np.array([dlogq_da, dlogq_db])*(-1)

def unif_gumbel2_log_posterior_neg_hessian(prior_param_vec, data):
    #log q (in the note)
    prior_alpha, prior_beta = prior_param_vec
    n = data.size
    negalpha_powered_data_times_squared_log_data_sum = np.sum((data**(-prior_alpha)) * (np.log(data)**2))
    negalpha_powered_data_times_log_data_sum = np.sum(data**(-prior_alpha) * np.log(data))

    h11 = - n/prior_alpha**2 - prior_beta * negalpha_powered_data_times_squared_log_data_sum
    h12 = negalpha_powered_data_times_log_data_sum
    h22 = -n/prior_beta**2

    return np.array([[h11, h12],[h12,h22]])*(-1)


import functools
newton_objective = functools.partial(unif_gumbel2_log_posterior_neg, data=gumbel2_smpl_55)
newton_gradient = functools.partial(unif_gumbel2_log_posterior_neg_gradient, data=gumbel2_smpl_55)
newton_hessian = functools.partial(unif_gumbel2_log_posterior_neg_hessian, data=gumbel2_smpl_55)

newton_inst1 = NewtonUnconstrained(
    newton_objective, 
    newton_gradient, 
    newton_hessian)
newton_inst1.run_newton_with_backtracking_line_search(np.array([5, 5]), method="cholesky")
# print(newton_inst1.get_minimizing_sequence())
# print(newton_inst1.get_decrement_sequence())
# print(newton_inst1.get_minimizing_function_value_sequence())


#laplace approximation
posterior_mode = newton_inst1.get_arg_min()
print("posterior_mode: ", posterior_mode)
laplace_approx_variance = np.linalg.inv(unif_gumbel2_log_posterior_neg_hessian(posterior_mode, gumbel2_smpl_55))
print("laplace_approx_covariance: \n",laplace_approx_variance)

laplace_approx_sample = rn_generator.multivariate_normal(posterior_mode, laplace_approx_variance, 10000)
plt.scatter(laplace_approx_sample[:,0], laplace_approx_sample[:,1]) 
plt.show()


