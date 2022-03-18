import csv
from random import normalvariate, gammavariate, seed, uniform
from math import sqrt, exp, log
from functools import partial

import numpy as np

from bayesian_tools.MCMC_Gibbs_Core import MCMC_Gibbs
from bayesian_tools.MCMC_MH_Core import MCMC_MH, MCMC_Diag

seed(20220317)

class PregData:
    def __init__(self):
        self._load()

    def _load(self, file_path = "pregnency-data.csv"):
        self.data = []
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            #header: "","patient","visit","time: t","weight: y"
            next(csv_reader)

            now_patient = 0
            for row in csv_reader:
                patient = int(row[1])
                time = float(row[3])
                weight = float(row[4])
                if now_patient != patient:
                    self.data.append([])
                    now_patient = patient
                self.data[-1].append((time, weight))
    
    def get_num_patient(self):
        return len(self.data)

    def get_num_total(self):
        return sum([len(patient_data) for patient_data in self.data])

    def _index_checker(self, i):
        if i==0 or i > self.get_num_patient():
            raise IndexError("index should be between 1 and " + str(self.get_num_patient()))

    def get_data_ith_patient(self, i):
        #start with 1!!
        self._index_checker(i)
        #(time: t, weight: y)
        return self.data[i-1]
    
    def get_visit_times_ith_patient(self, i):
        self._index_checker(i)
        return len(self.data[i-1])



pregdata_inst = PregData()
#for test
# print(pregdata_inst.get_data_ith_patient(1))
# print(pregdata_inst.get_data_ith_patient(30))
# print(pregdata_inst.get_num_patient())
# print(pregdata_inst.get_data_ith_patient(1))


class Gibbs_final(MCMC_Gibbs):
    def __init__(self, data_inst: PregData): #override
        self.MC_sample = []
        self.data_inst = data_inst
        self.hyperparam = {}

    def set_initial_parameter(self, beta_0i_vec, beta_1, beta_2, sigma2, beta_0_bar, tau2):
        if len(beta_0i_vec) != self.data_inst.get_num_patient():
            raise AttributeError("check the beta_0i_vec's dim")
        initial = [beta_0i_vec, beta_1, beta_2, sigma2, beta_0_bar, tau2]
        self.MC_sample.append(initial)

    def set_hyperparameter(self, mu, nu2, a_tau, b_tau, beta_1_bar, u2_1, beta_2_bar, u2_2, a_sigma, b_sigma):
        self.hyperparam = {
            "mu": mu, "nu2": nu2,
            "a_tau": a_tau, "b_tau": b_tau, 
            "beta_1_bar": beta_1_bar, "u2_1": u2_1,
            "beta_2_bar": beta_2_bar, "u2_2": u2_2,
            "a_sigma": a_sigma, "b_sigma": b_sigma
        }
        print("hyperparameter setting:", self.hyperparam)


    def full_conditional_sampler_beta_0i_vec(self, last_param):
        new_sample = [x for x in last_param]
        new_sample[0] = [x for x in last_param[0]]
        # new_sample
        #  0            1       2       3       4           5
        # [beta_0i_vec, beta_1, beta_2, sigma2, beta_0_bar, tau2]
        #update new

        for i_patient in range(1, self.data_inst.get_num_patient()+1):
            ty_tuple_list = self.data_inst.get_data_ith_patient(i_patient)
            sum_1_over_denom_squares = sum([1/((1 + new_sample[1]*exp(new_sample[2]*t))**2) for t,y in ty_tuple_list])
            sum_y_over_denom = sum([y/(1 + new_sample[1]*exp(new_sample[2]*t)) for t,y in ty_tuple_list])

            precision = sum_1_over_denom_squares/new_sample[3] + 1/new_sample[5]
            var = 1/precision
            mean = var * (sum_y_over_denom/new_sample[3] + new_sample[4]/new_sample[5])
            new_beta_0i = normalvariate(mean, sqrt(var))
            new_sample[0][i_patient-1] = new_beta_0i
        return new_sample


    def full_conditional_sampler_beta_1(self, last_param):
        new_sample = [x for x in last_param]
        # new_sample
        #  0            1       2       3       4           5
        # [beta_0i_vec, beta_1, beta_2, sigma2, beta_0_bar, tau2]
        #update new

        def beta_1_log_density_kernel(eval_pt, beta_0i_vec, beta_2, sigma_2, beta_1_bar, u2_1, data_inst: PregData):
            eval_pt = eval_pt[0]
            cum_sum = 0
            for i_patient in range(1, data_inst.get_num_patient()+1):
                ty_tuple_list = data_inst.get_data_ith_patient(i_patient)
                sum_beta0i_over_denom_squares = sum([beta_0i_vec[i_patient-1]**2 / (1 + eval_pt*exp(beta_2*t))**2 for (t,_) in ty_tuple_list])
                sum_ybeta0i_over_denom = sum([y * beta_0i_vec[i_patient-1] / (1 + eval_pt*exp(beta_2*t)) for (t,y) in ty_tuple_list])
                cum_sum += (sum_beta0i_over_denom_squares - 2*sum_ybeta0i_over_denom)

            log_kernel = -cum_sum/(2*sigma_2) - (eval_pt**2 - 2*eval_pt*beta_1_bar)/u2_1
            return log_kernel
                    
        def gaussian_sampler(last, sd):
            last = last[0]
            sample = normalvariate(last, sd)
            return [sample]

        def symmetric_log_density_kernel(from_smpl, to_smpl):
            return 0 #no need to implement
        
        beta_1_log_density_kernel_with_data = partial(beta_1_log_density_kernel, 
                                        beta_0i_vec = new_sample[0],
                                        beta_2 = new_sample[2],
                                        sigma_2 = new_sample[3],
                                        beta_1_bar = self.hyperparam["beta_1_bar"],
                                        u2_1 = self.hyperparam["u2_1"],
                                        data_inst = self.data_inst
                                        )
        gaussian_sampler_sd01 = partial(gaussian_sampler, sd=0.3) #0.3-0.5-1
        mcmc_mh_inst = MCMC_MH(beta_1_log_density_kernel_with_data, symmetric_log_density_kernel, gaussian_sampler_sd01, [new_sample[1]], random_seed=uniform(0,1))
        mcmc_mh_inst.generate_samples(2, verbose=False)
        new_beta_1 = mcmc_mh_inst.MC_sample[-1][0]
        new_sample[1] = new_beta_1
        return new_sample


    def full_conditional_sampler_beta_2(self, last_param):
        new_sample = [x for x in last_param]
        # new_sample
        #  0            1       2       3       4           5
        # [beta_0i_vec, beta_1, beta_2, sigma2, beta_0_bar, tau2]
        #update new

        def beta_2_log_density_kernel(eval_pt, beta_0i_vec, beta_1, sigma_2, beta_2_bar, u2_2, data_inst: PregData):
            eval_pt = eval_pt[0]
            cum_sum = 0
            for i_patient in range(1, data_inst.get_num_patient()+1):
                ty_tuple_list = data_inst.get_data_ith_patient(i_patient)
                sum_beta0i_over_denom_squares = sum([(beta_0i_vec[i_patient-1] / (1 + beta_1*exp(eval_pt*t)))**2 for (t,_) in ty_tuple_list])
                sum_ybeta0i_over_denom = sum([y * beta_0i_vec[i_patient-1] / (1 + beta_1*exp(eval_pt*t)) for (t,y) in ty_tuple_list])
                cum_sum += (sum_beta0i_over_denom_squares - 2*sum_ybeta0i_over_denom)

            log_kernel = -cum_sum/(2*sigma_2) - (eval_pt**2 - 2*eval_pt*beta_2_bar)/u2_2
            return log_kernel
                    
        def gaussian_sampler(last, sd):
            last = last[0]
            sample = normalvariate(last, sd)
            return [sample]

        def symmetric_log_density_kernel(from_smpl, to_smpl):
            return 0 #no need to implement
        
        beta_2_log_density_kernel_with_data = partial(beta_2_log_density_kernel, 
                                        beta_0i_vec = new_sample[0],
                                        beta_1 = new_sample[1],
                                        sigma_2 = new_sample[3],
                                        beta_2_bar = self.hyperparam["beta_2_bar"],
                                        u2_2 = self.hyperparam["u2_2"],
                                        data_inst = self.data_inst
                                        )
        gaussian_sampler_sd01 = partial(gaussian_sampler, sd=0.01) #0.01-0.05
        mcmc_mh_inst = MCMC_MH(beta_2_log_density_kernel_with_data, symmetric_log_density_kernel, gaussian_sampler_sd01, [new_sample[2]], random_seed=uniform(0,1))
        mcmc_mh_inst.generate_samples(5, verbose=False)
        new_beta_2 = mcmc_mh_inst.MC_sample[-1][0]

        new_sample[2] = new_beta_2
        return new_sample


    def full_conditional_sampler_sigma2(self, last_param):
        new_sample = [x for x in last_param]
        # new_sample
        #  0            1       2       3       4           5
        # [beta_0i_vec, beta_1, beta_2, sigma2, beta_0_bar, tau2]
        
        # keys: self.hyperparam
        # {"mu", "nu2", "a_tau", "b_tau", "beta_1_bar", "u2_1", "beta_2_bar", "u2_2","a_sigma", "b_sigma"
        
        #update new
        shape = self.hyperparam["a_sigma"] + self.data_inst.get_num_total()/2

        half_sum_of_squares = 0
        for i_patient in range(1, self.data_inst.get_num_patient()+1):
            for j_visit in self.data_inst.get_data_ith_patient(i_patient):
                t,y = j_visit
                squared = (y - new_sample[0][i_patient-1] / (1 + new_sample[1] * exp(new_sample[2] * t)))**2
                half_sum_of_squares += (squared/2)

        rate = self.hyperparam["b_sigma"] + half_sum_of_squares
        scale = 1/rate
        new_inv_sigma2 = gammavariate(shape, scale)
        new_sample[3] = 1/new_inv_sigma2
        return new_sample


    def full_conditional_sampler_beta_0_bar(self, last_param):
        new_sample = [x for x in last_param]
        # new_sample
        #  0            1       2       3       4           5
        # [beta_0i_vec, beta_1, beta_2, sigma2, beta_0_bar, tau2]
        
        # keys: self.hyperparam
        # {"mu", "nu2", "a_tau", "b_tau", "beta_1_bar", "u2_1", "beta_2_bar", "u2_2","a_sigma", "b_sigma"
        
        #update new
        precision = self.data_inst.get_num_patient()/new_sample[5] + 1/self.hyperparam["nu2"]
        var = 1/precision
        mean = var * (sum(new_sample[0]) / new_sample[5] + self.hyperparam["mu"]/self.hyperparam["nu2"])
        new_beta_0_bar = normalvariate(mean, sqrt(var))

        new_sample[4] = new_beta_0_bar
        return new_sample        


    def full_conditional_sampler_tau2(self, last_param):
        new_sample = [x for x in last_param]
        # new_sample
        #  0            1       2       3       4           5
        # [beta_0i_vec, beta_1, beta_2, sigma2, beta_0_bar, tau2]
        
        # keys: self.hyperparam
        # {"mu", "nu2", "a_tau", "b_tau", "beta_1_bar", "u2_1", "beta_2_bar", "u2_2","a_sigma", "b_sigma"
        
        #update new
        shape = self.hyperparam["a_tau"] + self.data_inst.get_num_patient()/2
        rate = self.hyperparam["b_tau"] + sum([(b0i-new_sample[4])**2 for b0i in new_sample[0]])/2
        scale = 1/rate
        new_inv_tau2 = gammavariate(shape, scale)

        new_sample[5] = 1/new_inv_tau2
        return new_sample


    def gibbs_sampler(self): #override
        last = self.MC_sample[-1]
        new = [x for x in last]
        #update new
        new = self.full_conditional_sampler_beta_0i_vec(new)
        new = self.full_conditional_sampler_beta_1(new)
        new = self.full_conditional_sampler_beta_2(new)
        new = self.full_conditional_sampler_sigma2(new)
        new = self.full_conditional_sampler_beta_0_bar(new)
        new = self.full_conditional_sampler_tau2(new)
        self.MC_sample.append(new)


gibbs_final_inst1 = Gibbs_final(pregdata_inst)
# def set_initial_parameter(self, beta_0i_vec, beta_1, beta_2, sigma2, beta_0_bar, tau2):
gibbs_final_inst1.set_initial_parameter(
    [0 for _ in range(pregdata_inst.get_num_patient())], 0, 0, 1, 0, 1)

# def set_hyperparameter(self, mu, nu2, a_tau, b_tau, beta_1_bar, u2_1, beta_2_bar, u2_2, a_sigma, b_sigma):
gibbs_final_inst1.set_hyperparameter(0, 100, 0.1, 0.1, 0, 100, 0, 100, 0.1, 0.1)

gibbs_final_inst1.generate_samples(300000)
### running info ###
# hyperparameter setting: {'mu': 0, 'nu2': 100, 'a_tau': 0.1, 'b_tau': 0.1, 'beta_1_bar': 0, 'u2_1': 100, 'beta_2_bar': 0, 'u2_2': 100, 'a_sigma': 0.1, 'b_sigma': 0.1}
# estimated running time:  11.0 min  25.509681701660156 sec
# iteration 500 / 300000
# iteration 1000 / 300000
# ...
# iteration 299500 / 300000
# iteration 300000 / 300000  done! (elapsed time for execution:  13.0 min  15.209720611572266 sec)



#  0            1       2       3       4           5
# [beta_0i_vec, beta_1, beta_2, sigma2, beta_0_bar, tau2]
mc_samples_part1 = [sample[1:] for sample in gibbs_final_inst1.MC_sample]
mc_samples_part2 = [sample[0] for sample in gibbs_final_inst1.MC_sample]


gibbs_p5_diag_inst1_part1 = MCMC_Diag()
gibbs_p5_diag_inst1_part1.set_mc_samples_from_list(mc_samples_part1)
gibbs_p5_diag_inst1_part1.write_samples("part1")
gibbs_p5_diag_inst1_part1.set_variable_names(["beta_1", "beta_2", "sigma2", "beta_0_bar", "tau2"])
gibbs_p5_diag_inst1_part1.show_traceplot((2,3))


gibbs_p5_diag_inst1_part2 = MCMC_Diag()
gibbs_p5_diag_inst1_part2.set_mc_samples_from_list(mc_samples_part2)
gibbs_p5_diag_inst1_part2.write_samples("part2")
gibbs_p5_diag_inst1_part2.set_variable_names(["beta_0"+str(i) for i in range(1,31)])
gibbs_p5_diag_inst1_part2.show_traceplot((2,3), choose_dims=[0,1,2,3,4,5])
