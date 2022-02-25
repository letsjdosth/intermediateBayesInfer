from math import log, sqrt, exp
from random import gammavariate, normalvariate
from functools import partial

from MCMC_MH_Core import MCMC_MH, MCMC_Diag

# For Z
def inv_gaussian_log_density_kernel(eval_pt, param_vec):
    theta1 = param_vec[0]
    theta2 = param_vec[1]
    eval_pt = eval_pt[0]
    log_kernel = -(3/2)*log(eval_pt) 
    log_kernel -= (theta1*eval_pt + theta2/eval_pt )
    log_kernel += (2*sqrt(theta1*theta2) + (1/2)*log(2*theta2))
    return log_kernel

def gamma_sampler(last, shape, rate):
    #last: not used in indep.MCMC
    sample = gammavariate(shape, 1/rate)
    return [sample]

def gamma_log_density_kernel(from_smpl, to_smpl, shape, rate):
    #from_smpl: not used in indep.MCMC
    to_smpl = to_smpl[0]
    log_kernel = (shape-1) * log(to_smpl) - rate*to_smpl
    return log_kernel

gamma_sampler_s3_r4 = partial(gamma_sampler, shape=3, rate=4)
gamma_log_density_kernel_s3_r4 = partial(gamma_log_density_kernel, shape=3, rate=4)
inv_gaussian_log_density_kernel_1p5_2 = partial(inv_gaussian_log_density_kernel, param_vec=(1.5, 2))

inst_p1a = MCMC_MH(inv_gaussian_log_density_kernel_1p5_2, gamma_log_density_kernel_s3_r4, gamma_sampler_s3_r4, [1], random_seed=20220224)
inst_p1a.generate_samples(50000)
inst_p1a_diag = MCMC_Diag()
inst_p1a_diag.set_mc_sample_from_MCMC_MH(inst_p1a)
inst_p1a_diag.show_traceplot((1,1))
inst_p1a_diag.show_acf(20, (1,1))
inst_p1a_diag.burnin(3000)
inst_p1a_diag.thinning(10)

inst_p1a_diag.show_traceplot((1,1))
inst_p1a_diag.show_hist((1,1))
inst_p1a_diag.show_acf(20, (1,1))
print("mean of z:", inst_p1a_diag.get_sample_mean()) #true: 1.1547
print("var of z:", inst_p1a_diag.get_sample_var())
print("0.95 cred.int. of z: ", inst_p1a_diag.get_sample_quantile([0.025, 0.975]))

#invert to 1/Z
inst_p1a_diag_inverted = MCMC_Diag()
inverted_MC_sample = [[1/sample[0]] for sample in inst_p1a.MC_sample]
inst_p1a_diag_inverted.set_mc_samples_from_list(inverted_MC_sample)
print("mean of 1/z: ", inst_p1a_diag_inverted.get_sample_mean()) #true: 1.1160
print("var of 1/z: ", inst_p1a_diag_inverted.get_sample_var())
print("0.95 cred.int. of 1/z: ", inst_p1a_diag_inverted.get_sample_quantile([0.025, 0.975]))


#for W
def log_inv_gaussian_log_density_kernel(eval_pt, param_vec):
    theta1 = param_vec[0]
    theta2 = param_vec[1]
    eval_pt = eval_pt[0]
    log_kernel = -0.5*eval_pt
    log_kernel -= (theta1*exp(eval_pt) + theta2/exp(eval_pt))
    log_kernel += (2*sqrt(theta1*theta2) + (1/2)*log(2*theta2))
    return log_kernel

def gaussian_sampler(last, sigma):
    #sigma: sd
    last = last[0]
    sample = normalvariate(last, sigma)
    return [sample]

def symmetric_log_density_kernel(from_smpl, to_smpl):
    return 0 #no need to implement

gaussian_sampler_sd1 = partial(gaussian_sampler, sigma=1)
log_inv_gaussian_log_density_kernel_1p5_2 = partial(log_inv_gaussian_log_density_kernel, param_vec=(1.5, 2))

inst_p1b = MCMC_MH(log_inv_gaussian_log_density_kernel_1p5_2, symmetric_log_density_kernel, gaussian_sampler_sd1, [1], random_seed=20220224+1)
inst_p1b.generate_samples(50000)
inst_p1b_diag = MCMC_Diag()
inst_p1b_diag.set_mc_sample_from_MCMC_MH(inst_p1b)
inst_p1b_diag.show_traceplot((1,1))
inst_p1b_diag.show_acf(20, (1,1))
inst_p1b_diag.burnin(3000)
inst_p1b_diag.thinning(4)

inst_p1b_diag.show_traceplot((1,1))
inst_p1b_diag.show_hist((1,1))
inst_p1b_diag.show_acf(20, (1,1))
print("mean of w:", inst_p1b_diag.get_sample_mean())
print("var of w:", inst_p1b_diag.get_sample_var())
print("0.95 cred.int. of w: ", inst_p1b_diag.get_sample_quantile([0.025, 0.975]))

#invert to Z
inst_p1b_diag_z = MCMC_Diag()
z_MC_sample = [[exp(sample[0])] for sample in inst_p1b.MC_sample]
inst_p1b_diag_z.set_mc_samples_from_list(z_MC_sample)
inst_p1b_diag_z.show_hist((1,1))
print("mean of z: ", inst_p1b_diag_z.get_sample_mean()) #true: 1.1547
print("var of z: ", inst_p1b_diag_z.get_sample_var())
print("0.95 cred.int. of z: ", inst_p1b_diag_z.get_sample_quantile([0.025, 0.975]))
