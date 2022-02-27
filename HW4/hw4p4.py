from random import gammavariate, normalvariate, uniform, seed, randint
from math import log, exp
from functools import partial

import numpy as np

from MCMC_MH_Core import MCMC_MH, MCMC_Diag
from MCMC_Gibbs_Core import MCMC_Gibbs
from newton import NewtonUnconstrained

##################################################################
## DATA:
## Carlin, Gelfand and Smith (1992): Hierarchical Bayesian Analysis of Changepoint Problems
## Counts of coal mining disasters in Great Britain by year from 1851 to 1962.
##################################################################
mining_data = [
    4,5,4,1,0,4,3,4,0,6,3,3,4,0,2,6,3,3,5,4,5,3,1,4,4,1,5,5,3,
    4,2,5,2,2,3,4,2,1,3,2,2,1,1,1,1,3,0,0,1,0,1,1,0,0,3,1,0,3,
    2,2,0,1,1,1,0,1,0,1,0,0,0,2,1,0,0,0,1,1,0,2,3,3,1,1,2,1,1,
    1,1,2,4,2,0,0,0,1,4,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1
    ]
print(len(mining_data)) #112

seed(20220227)

class Gibbs_with_MH_p4(MCMC_Gibbs):
    def __init__(self, initial, data, hyper_param): #override
        self.MC_sample = [initial]
        self.data = data
        self.n = len(data)
        
        #hyper_param: [alpha, beta, gamma, delta]
        self.hyper_param =  hyper_param
    
    def get_data_sum(self):
        if not hasattr(self, 'data_sum'):
            self.data_sum = sum(self.data)
        return self.data_sum

    def full_conditional_sampler_theta(self, last_param):
        #last_param: [theta, phi, m]
        #hyper_param: [alpha, beta, gamma, delta]
        alpha = self.hyper_param[0]
        beta = self.hyper_param[1]
        m = last_param[2]
        shape = alpha + sum(self.data[:m])
        rate = beta + m
        new_theta = gammavariate(shape, 1/rate)
        
        new_sample = [x for x in last_param]
        new_sample[0] = new_theta
        return new_sample


    def full_conditional_sampler_phi(self, last_param):
        #last_param: [theta, phi, m]
        #hyper_param: [alpha, beta, gamma, delta]
        gamma = self.hyper_param[2]
        delta = self.hyper_param[3]
        m = last_param[2]
        shape = gamma + sum(self.data[m:])
        rate = delta + self.n - m
        new_phi = gammavariate(shape, 1/rate)
        
        new_sample = [x for x in last_param]
        new_sample[1] = new_phi
        return new_sample

    def full_conditional_sampler_m(self, last_param):
        #last_param: [theta, phi, m]
        #hyper_param: [alpha, beta, gamma, delta]
        m = last_param[2]

        def log_target_density_kernel(eval_pt, last_param, data):
            eval_pt = int(eval_pt[0])
            theta = last_param[0]
            phi = last_param[1]

            first_m_data_sum = sum(data[:eval_pt])
            last_data_sum = sum(data[eval_pt:])

            log_kernel = first_m_data_sum*log(theta) + last_data_sum*log(phi) - eval_pt*theta + eval_pt+phi
            return log_kernel

        def log_proposal_density(from_smpl, to_smpl, n, window):
            if from_smpl[0] - window < 0:
                log_density = -log(from_smpl[0] + window + 1)
            if from_smpl[0] + window > n:
                log_density = -log(n - from_smpl[0] + window + 1)
            else:
                log_density = -log(window*2 + 1)
            return log_density
            
        def proposal_sampler(last, n, window):
            return [randint(max(0, last[0]-window), min(n, last[0]+window))]

        
        log_target_density_kernel_with_data = partial(log_target_density_kernel, last_param=last_param, data=self.data)
        proposal_sampler_with_data = partial(proposal_sampler, n=self.n, window=10)
        log_proposal_density_with_data = partial(log_proposal_density, n=self.n, window=10)
        mcmc_mh_inst = MCMC_MH(log_target_density_kernel_with_data, log_proposal_density_with_data, proposal_sampler_with_data, [m], random_seed=uniform(0,1))
        mcmc_mh_inst.generate_samples(1, verbose=False)
        new_m = mcmc_mh_inst.MC_sample[-1][0]
        
        new_sample = [x for x in last_param]
        new_sample[2] = new_m
        return new_sample

    def gibbs_sampler(self): #override
        last = self.MC_sample[-1]
        new = [x for x in last] #[theta, phi, m]

        #update new
        new = self.full_conditional_sampler_theta(new)
        new = self.full_conditional_sampler_phi(new)
        new = self.full_conditional_sampler_m(new)
        self.MC_sample.append(new)


hyper_param_set1 = [0.01, 0.01, 0.01, 0.01] #hyper_param: [alpha, beta, gamma, delta]
gibbs_inst_1 = Gibbs_with_MH_p4([2,2,10], mining_data, hyper_param_set1)
gibbs_inst_1.generate_samples(100000)

gibbs_diag_inst_1 = MCMC_Diag()
gibbs_diag_inst_1.set_mc_sample_from_MCMC_MH(gibbs_inst_1)
# gibbs_diag_inst_1.burnin(3000)
gibbs_diag_inst_1.show_traceplot((3,1))
gibbs_diag_inst_1.show_hist((3,1))
# gibbs_diag_inst_1.show_scatterplot(0,1)
gibbs_diag_inst_1.show_acf(30, (3,1))


#1000은 너무한듯 ㅋㅋ

hyper_param_set2 = [0.01, 0.01, 1000, 1000] #hyper_param: [alpha, beta, gamma, delta]
gibbs_inst_2 = Gibbs_with_MH_p4([2,2,10], mining_data, hyper_param_set2)
gibbs_inst_2.generate_samples(100000)

gibbs_diag_inst_2 = MCMC_Diag()
gibbs_diag_inst_2.set_mc_sample_from_MCMC_MH(gibbs_inst_2)
# gibbs_diag_inst_2.burnin(3000)
gibbs_diag_inst_2.show_traceplot((3,1))
gibbs_diag_inst_2.show_hist((3,1))
# gibbs_diag_inst_2.show_scatterplot(0,1)
gibbs_diag_inst_2.show_acf(30, (3,1))



hyper_param_set3 = [1000, 1000, 0.01, 0.01] #hyper_param: [alpha, beta, gamma, delta]
gibbs_inst_3 = Gibbs_with_MH_p4([2,2,10], mining_data, hyper_param_set3)
gibbs_inst_3.generate_samples(100000)

gibbs_diag_inst_3 = MCMC_Diag()
gibbs_diag_inst_3.set_mc_sample_from_MCMC_MH(gibbs_inst_3)
# gibbs_diag_inst_3.burnin(3000)
gibbs_diag_inst_3.show_traceplot((3,1))
gibbs_diag_inst_3.show_hist((3,1))
# gibbs_diag_inst_3.show_scatterplot(0,1)
gibbs_diag_inst_3.show_acf(30, (3,1))




hyper_param_set4 = [1000, 1000, 1000, 1000] #hyper_param: [alpha, beta, gamma, delta]
gibbs_inst_4 = Gibbs_with_MH_p4([2,2,10], mining_data, hyper_param_set4)
gibbs_inst_4.generate_samples(100000)

gibbs_diag_inst_4 = MCMC_Diag()
gibbs_diag_inst_4.set_mc_sample_from_MCMC_MH(gibbs_inst_4)
# gibbs_diag_inst_4.burnin(3000)
gibbs_diag_inst_4.show_traceplot((3,1))
gibbs_diag_inst_4.show_hist((3,1))
# gibbs_diag_inst_4.show_scatterplot(0,1)
gibbs_diag_inst_4.show_acf(30, (3,1))

