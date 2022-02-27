from random import gammavariate, normalvariate, uniform, seed
from math import lgamma, log, exp
from functools import partial

import numpy as np
from scipy.special import polygamma

from MCMC_MH_Core import MCMC_MH, MCMC_Diag
from MCMC_Gibbs_Core import MCMC_Gibbs
from newton import NewtonUnconstrained



mydata = []
with open("HW4/my-data.txt") as f:
    for line in f:
        mydata.append(float(line))
print(mydata)
data_log_sum = sum([log(x) for x in mydata])
data_sum = sum(mydata)

seed(20220226)


class Gibbs_with_MH(MCMC_Gibbs):
    def __init__(self, data, initial): #override
        self.data = data
        self.n = len(data)
        self.MC_sample = [initial]

    def get_data_sum(self):
        if not hasattr(self, 'data_sum'):
            self.data_sum = sum(self.data)
        return self.data_sum

    
    def get_data_log_sum(self):
        if not hasattr(self, 'data_log_sum'):
            self.data_log_sum = sum([log(x) for x in self.data])
        return self.data_log_sum

    # full-conditionals implementation

    def full_conditional_sampler_theta(self, last_param):
        nu = last_param[0]
        # theta = last_param[1] #not used
        shape = self.n * nu + 2 
        rate = self.get_data_sum() + 2
        new_theta = gammavariate(shape, 1/rate)
        return [nu, new_theta]

    def full_conditional_sampler_log_nu(self, last_param):
        nu = last_param[0]
        log_nu = log(nu)
        theta = last_param[1]

        def log_log_nu_density_kernel(eval_pt, theta, data_log_sum, data_num):
            log_nu = eval_pt[0]
            nu = exp(log_nu)
            log_kernel = (data_num*nu)*log(theta) - (data_num)*lgamma(nu) + (nu-1)*data_log_sum + 3*log_nu - nu
            return log_kernel
                    
        def gaussian_sampler(last, sd):
            last = last[0]
            sample = normalvariate(last, sd)
            return [sample]

        def symmetric_log_density_kernel(from_smpl, to_smpl):
            return 0 #no need to implement
        
        log_log_nu_density_kernel_for_log_nu = partial(log_log_nu_density_kernel, theta=theta, data_log_sum=self.get_data_log_sum(), data_num=self.n)
        gaussian_sampler_sd1 = partial(gaussian_sampler, sd=0.1)
        mcmc_mh_inst = MCMC_MH(log_log_nu_density_kernel_for_log_nu, symmetric_log_density_kernel, gaussian_sampler_sd1, [log_nu], random_seed=uniform(0,1))
        mcmc_mh_inst.generate_samples(2, verbose=False)
        new_log_nu = mcmc_mh_inst.MC_sample[-1]
        return [exp(new_log_nu[0]), theta]

    # main-sampler
    
    def gibbs_sampler(self): #override
        last = self.MC_sample[-1]
        new = [x for x in last] #[nu, theta]
        new = self.full_conditional_sampler_log_nu(new)
        new = self.full_conditional_sampler_theta(new)
        self.MC_sample.append(new)


#Gibbs

gibbs_inst = Gibbs_with_MH(mydata, [1,1])
gibbs_inst.generate_samples(30000)

gibbs_diag_inst = MCMC_Diag()
gibbs_diag_inst.set_mc_sample_from_MCMC_MH(gibbs_inst)
gibbs_diag_inst.burnin(3000)
gibbs_diag_inst.show_traceplot((1,2))
gibbs_diag_inst.show_hist((1,2))
gibbs_diag_inst.show_scatterplot(0,1)
gibbs_diag_inst.show_acf(30, (1,2))


# MH

def joint_log_density_kernel(eval_pt, data_num, data_log_sum, data_sum):
    w = eval_pt[0]
    t = eval_pt[1]
    log_kernel = t*data_num*exp(w) - data_num*lgamma(exp(w))
    log_kernel += (exp(w)-1)*data_log_sum
    log_kernel -= exp(t)*data_sum
    log_kernel += (2*w - exp(w) + t - 2*exp(t) + w + t)
    return log_kernel

def gaussian_sampler_2d(last, sd):
    sample0 = normalvariate(last[0], sd)
    sample1 = normalvariate(last[1], sd)
    return [sample0, sample1]

def symmetric_log_density_kernel(from_smpl, to_smpl):
    return 0 #no need to implement

joint_log_density_kernel_with_data = partial(joint_log_density_kernel, data_num=len(mydata), data_log_sum=data_log_sum, data_sum=data_sum)
gaussian_sampler_2d_with_sd = partial(gaussian_sampler_2d, sd=0.1)
mh_inst = MCMC_MH(joint_log_density_kernel_with_data, symmetric_log_density_kernel, gaussian_sampler_2d_with_sd, [1,1], 20220226)
mh_inst.generate_samples(30000)

# mh_diag_inst = MCMC_Diag()
# mh_diag_inst.set_mc_sample_from_MCMC_MH(mh_inst)
# mh_diag_inst.show_traceplot((1,2))
# mh_diag_inst.show_hist((1,2))
# mh_diag_inst.show_scatterplot(0,1)

mh_diag_inst2 = MCMC_Diag()
nu_theta_samples = [[exp(sample[0]), exp(sample[1])] for sample in mh_inst.MC_sample]
mh_diag_inst2.set_mc_samples_from_list(nu_theta_samples)
mh_diag_inst2.burnin(3000)
mh_diag_inst2.show_traceplot((1,2))
mh_diag_inst2.show_hist((1,2))
mh_diag_inst2.show_scatterplot(0,1)
mh_diag_inst2.show_acf(30, (1,2))


#MH - laplace approximation proposal

def joint_log_density_kernel_neg(eval_pt, data_num, data_log_sum, data_sum):
    return joint_log_density_kernel(eval_pt, data_num, data_log_sum, data_sum)*(-1)

def joint_log_density_kernel_neg_gradient(eval_pt, data_num, data_log_sum, data_sum):
    w = eval_pt[0]
    nu = exp(w)
    t = eval_pt[1]

    dq_dw = t*data_num*nu - data_num*polygamma(0, nu)*nu + nu*data_log_sum + 3 - nu
    dq_dt = data_num*nu - exp(t)*data_sum + 2 - 2*exp(t)
    return np.array([dq_dw, dq_dt])*(-1)

def joint_log_density_kernel_neg_hessian(eval_pt, data_num, data_log_sum, data_sum):
    w = eval_pt[0]
    nu = exp(w)
    t = eval_pt[1]

    d2q_dw2 = t*data_num*nu + data_num*(polygamma(0,nu)**2)*(nu**2)
    d2q_dw2 -= data_num*(polygamma(1,nu) + polygamma(0,nu)**2)*nu**2
    d2q_dw2 -= data_num*(polygamma(0,nu))*nu
    d2q_dw2 += (nu*data_log_sum - nu)

    d2q_dwdt = data_num*nu
    d2q_d2t = -exp(t)*(data_sum+2)
    return np.array([[d2q_dw2, d2q_dwdt],[d2q_dwdt, d2q_d2t]])*(-1)

newton_objective = partial(joint_log_density_kernel_neg, data_num=len(mydata), data_log_sum=data_log_sum, data_sum=data_sum)
newton_gradient = partial(joint_log_density_kernel_neg_gradient, data_num=len(mydata), data_log_sum=data_log_sum, data_sum=data_sum)
newton_hessian = partial(joint_log_density_kernel_neg_hessian, data_num=len(mydata), data_log_sum=data_log_sum, data_sum=data_sum)
newton_inst = NewtonUnconstrained(newton_objective, newton_gradient, newton_hessian)

newton_inst.run_newton_with_backtracking_line_search(np.array([1, 0.1]), method="cholesky")
mode = newton_inst.get_arg_min()
print("laplace approximation - mode:", mode)
laplace_approx_cov = np.linalg.inv(newton_hessian(mode))
print("laplace approximation - cov matrix:\n", laplace_approx_cov)

def multivariate_normal_sampler(last, mean, cov):
    #last: not used (independent proposal)
    return np.random.multivariate_normal(mean, cov)
laplace_approxed_sampler = partial(multivariate_normal_sampler, mean=mode, cov=laplace_approx_cov)

def log_multivariate_normal_density(from_smpl, to_smpl, mean, inv_cov):
    #from_smpl: not used (independent proposal)
    to_smpl = np.array(to_smpl)
    kernel = -0.5 * (to_smpl-mean).transpose() @ inv_cov @ (to_smpl-mean)
    return kernel

laplace_approxed_log_density = partial(log_multivariate_normal_density, mean=mode, inv_cov = newton_hessian(mode))
mh_laplace_inst = MCMC_MH(joint_log_density_kernel_with_data, laplace_approxed_log_density, laplace_approxed_sampler, mode, 20220227+10)
mh_laplace_inst.generate_samples(30000)

# mh_laplace_diag_inst = MCMC_Diag()
# mh_laplace_diag_inst.set_mc_sample_from_MCMC_MH(mh_laplace_inst)
# mh_laplace_diag_inst.burnin(3000)
# mh_laplace_diag_inst.show_traceplot((1,2))
# mh_laplace_diag_inst.show_hist((1,2))

mh_laplace_diag_inst2 = MCMC_Diag()
laplace_nu_theta_samples = [[exp(sample[0]), exp(sample[1])] for sample in mh_laplace_inst.MC_sample]
mh_laplace_diag_inst2.set_mc_samples_from_list(laplace_nu_theta_samples)
mh_laplace_diag_inst2.burnin(3000)
mh_laplace_diag_inst2.show_traceplot((1,2))
mh_laplace_diag_inst2.show_hist((1,2))
mh_laplace_diag_inst2.show_scatterplot(0,1)
mh_laplace_diag_inst2.show_acf(30, (1,2))


print("gibbs:   mean:", gibbs_diag_inst.get_sample_mean())
print("gibbs: 0.95CI:", gibbs_diag_inst.get_sample_quantile([0.025, 0.5, 0.975]))

print("MH:   mean:", mh_diag_inst2.get_sample_mean())
print("MH: 0.95CI:", mh_diag_inst2.get_sample_quantile([0.025, 0.5, 0.975]))

print("MH_laplace:   mean:", mh_laplace_diag_inst2.get_sample_mean())
print("MH_laplace: 0.95CI:", mh_laplace_diag_inst2.get_sample_quantile([0.025, 0.5, 0.975]))
