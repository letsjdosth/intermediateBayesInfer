import csv
from random import normalvariate, gammavariate, seed
from math import sqrt

import numpy as np

from MCMC_Gibbs_Core import MCMC_Gibbs
from MCMC_MH_Core import MCMC_Diag

seed(20220302)

class SouzaData:
    def __init__(self):
        self._load()

    def _load(self, file_path = "HW4/Q5-madeup-data.csv"):
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
        self._index_checker(i)
        #(time: t, weight: y)
        return self.data[i-1]
    
    def get_visit_times_ith_patient(self, i):
        self._index_checker(i)
        return len(self.data[i-1])

    def get_sum_vec_ith_patient(self, i):
        self._index_checker(i)
        ith_array_data = self.get_data_ith_patient(i)
        t_vec = [item[0] for item in ith_array_data]
        y_vec = [item[1] for item in ith_array_data]
        return (sum(t_vec), sum(y_vec))
    
    def get_yt_product_sum_ith_patient(self, i):
        self._index_checker(i)
        ith_array_data = np.array(self.get_data_ith_patient(i))
        product_vec = [item[0]*item[1] for item in ith_array_data]
        return sum(product_vec)

    def get_t_square_sum_vec_ith_patient(self, i):
        self._index_checker(i)
        ith_array_data = np.array(self.get_data_ith_patient(i))
        squared_vec = [item[0]**2 for item in ith_array_data]
        return sum(squared_vec)


#data load & check
data_inst = SouzaData()
# print(data_inst.get_data_ith_patient(10))
# print(data_inst.get_num_patient())
# print(data_inst.get_visit_times_ith_patient(10))
# print(data_inst.get_sum_vec_ith_patient(10))
# print(data_inst.get_num_total())
# print(data_inst.get_yt_product_sum_ith_patient(10))
# print(data_inst.get_t_square_sum_vec_ith_patient(10))

class Gibb_p5(MCMC_Gibbs):
    def __init__(self, data_inst: SouzaData): #override
        self.MC_sample = []
        self.data_inst = data_inst
        self.hyperparam = {}

    def set_initial_parameter(self, alpha, beta, tau_alpha, tau_beta, sigma_square, alpha_i_vec, beta_i_vec):
        if len(alpha_i_vec) != self.data_inst.get_num_patient():
            raise AttributeError("check the alpha_i_vec's dim")
        if len(beta_i_vec) != self.data_inst.get_num_patient():
            raise AttributeError("check the beta_i_vec's dim")
        initial = [alpha, beta, tau_alpha, tau_beta, sigma_square, alpha_i_vec, beta_i_vec]
        self.MC_sample.append(initial)

    def set_hyperparameter(self, a_sigma, b_sigma, a_alpha, b_alpha, a_beta, b_beta, P2_alpha, P2_beta):
        self.hyperparam = {
            "a_sigma": a_sigma, "b_sigma": b_sigma, 
            "a_alpha": a_alpha, "b_alpha": b_alpha, 
            "a_beta": a_beta, "b_beta": b_beta, 
            "P2_alpha": P2_alpha, "P2_beta":P2_beta
        }
        print("hyperparameter setting:", self.hyperparam)

    def full_conditional_sampler_alpha(self, last_param):
        new_sample = [x for x in last_param]
        #  0      1     2          3         4             5            6
        # [alpha, beta, tau_alpha, tau_beta, sigma_square, alpha_i_vec, beta_i_vec]
        #update new
        precision = self.data_inst.get_num_patient()/new_sample[2] + 1/self.hyperparam["P2_alpha"]
        var = 1/precision
        mean = var * sum(new_sample[5])/new_sample[2]
        new_alpha = normalvariate(mean, sqrt(var))

        new_sample[0] = new_alpha
        return new_sample

    def full_conditional_sampler_beta(self, last_param):
        new_sample = [x for x in last_param]
        #  0      1     2          3         4             5            6
        # [alpha, beta, tau_alpha, tau_beta, sigma_square, alpha_i_vec, beta_i_vec]
        #update new
        precision = self.data_inst.get_num_patient()/new_sample[3] + 1/self.hyperparam["P2_beta"]
        var = 1/precision
        mean = var * sum(new_sample[6])/new_sample[3]
        new_beta = normalvariate(mean, sqrt(var))

        new_sample[1] = new_beta
        return new_sample        


    def full_conditional_sampler_tau_a(self, last_param):
        new_sample = [x for x in last_param]
        #  0      1     2          3         4             5            6
        # [alpha, beta, tau_alpha, tau_beta, sigma_square, alpha_i_vec, beta_i_vec]
        #update new
        shape = self.hyperparam["a_alpha"] + self.data_inst.get_num_patient()/2
        rate = self.hyperparam["b_alpha"] + sum([(ai-new_sample[0])**2 for ai in new_sample[5]])/2
        scale = 1/rate
        # if scale == 0:
        #     scale = 1e-16
        new_inv_tau_alpha = gammavariate(shape, scale)

        new_sample[2] = 1/new_inv_tau_alpha
        return new_sample

    def full_conditional_sampler_tau_b(self, last_param):
        new_sample = [x for x in last_param]
        #  0      1     2          3         4             5            6
        # [alpha, beta, tau_alpha, tau_beta, sigma_square, alpha_i_vec, beta_i_vec]
        #update new
        shape = self.hyperparam["a_beta"] + self.data_inst.get_num_patient()/2
        rate = self.hyperparam["b_beta"] + sum([(bi-new_sample[1])**2 for bi in new_sample[6]])/2
        scale = 1/rate
        # if scale == 0:
        #     scale = 1e-16
        new_inv_tau_beta = gammavariate(shape, scale)

        new_sample[3] = 1/new_inv_tau_beta
        return new_sample

    def full_conditional_sampler_sigma2(self, last_param):
        new_sample = [x for x in last_param]
        #  0      1     2          3         4             5            6
        # [alpha, beta, tau_alpha, tau_beta, sigma_square, alpha_i_vec, beta_i_vec]
        #update new
        shape = self.hyperparam["a_sigma"] + self.data_inst.get_num_total()/2
        half_sum_of_squares = 0
        for i_patient in range(1, self.data_inst.get_num_patient()+1):
            for j_visit in self.data_inst.get_data_ith_patient(i_patient):
                t,y = j_visit
                squared = (y - new_sample[5][i_patient-1] - new_sample[6][i_patient-1] * t)**2
                half_sum_of_squares += (squared/2)

        rate = self.hyperparam["b_sigma"] + half_sum_of_squares
        scale = 1/rate
        # if scale == 0:
        #     scale = 1e-16
        new_inv_sigma_square = gammavariate(shape, scale)
        new_sample[4] = 1/new_inv_sigma_square
        return new_sample

    def full_conditional_sampler_a_i(self, last_param):
        new_sample = [x for x in last_param]
        new_sample[5] = [x for x in last_param[5]]
        #  0      1     2          3         4             5            6
        # [alpha, beta, tau_alpha, tau_beta, sigma_square, alpha_i_vec, beta_i_vec]
        #update new
        for i_patient in range(1, self.data_inst.get_num_patient()+1):
            t_sum, y_sum = self.data_inst.get_sum_vec_ith_patient(i_patient)

            precision = self.data_inst.get_visit_times_ith_patient(i_patient)/new_sample[4] + 1/new_sample[2]
            var = 1/precision
            mean = var * ((y_sum - new_sample[6][i_patient-1] * t_sum)/new_sample[4] + new_sample[0]/new_sample[2])
            new_alpha_i = normalvariate(mean, sqrt(var))
            new_sample[5][i_patient-1] = new_alpha_i
        return new_sample

    def full_conditional_sampler_b_i(self, last_param):
        new_sample = [x for x in last_param]
        new_sample[6] = [x for x in last_param[6]]
        #  0      1     2          3         4             5            6
        # [alpha, beta, tau_alpha, tau_beta, sigma_square, alpha_i_vec, beta_i_vec]
        #update new
        for i_patient in range(1, self.data_inst.get_num_patient()+1):
            t_sum, y_sum = self.data_inst.get_sum_vec_ith_patient(i_patient)
            yt_sum = self.data_inst.get_yt_product_sum_ith_patient(i_patient)
            t2_sum = self.data_inst.get_t_square_sum_vec_ith_patient(i_patient)
            # n_i = self.data_inst.get_visit_times_ith_patient(i_patient)

            precision = t2_sum/new_sample[4] + 1/new_sample[3]
            var = 1/precision
            mean = var * ((yt_sum - new_sample[5][i_patient-1] * t_sum)/new_sample[4] + new_sample[1]/new_sample[3])
            new_beta_i = normalvariate(mean, sqrt(var))
            new_sample[6][i_patient-1] = new_beta_i
        return new_sample


    def gibbs_sampler(self): #override
        last = self.MC_sample[-1]
        new = [x for x in last]
        #update new
        
        new = self.full_conditional_sampler_a_i(new)
        new = self.full_conditional_sampler_b_i(new)
        new = self.full_conditional_sampler_tau_a(new)
        new = self.full_conditional_sampler_tau_b(new)
        new = self.full_conditional_sampler_alpha(new)
        new = self.full_conditional_sampler_beta(new)
        new = self.full_conditional_sampler_sigma2(new)
        self.MC_sample.append(new)


gibbs_p5_inst1 = Gibb_p5(data_inst)
# gibbs_p5_inst1.set_hyperparameter(1, 0.1, 1, 0.1, 1, 0.1, 1, 1)
gibbs_p5_inst1.set_hyperparameter(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
gibbs_p5_inst1.set_initial_parameter(0, 0, 1, 1, 1, [0 for _ in range(data_inst.get_num_patient())], [0 for _ in range(data_inst.get_num_patient())])
gibbs_p5_inst1.generate_samples(30000)
# gibbs_p5_inst1.write_samples("HW4/hw4p5_samples")


mc_samples_part1 = [sample[0:5] for sample in gibbs_p5_inst1.MC_sample]
mc_samples_part2 = [sample[5][0:6] for sample in gibbs_p5_inst1.MC_sample]
mc_samples_part3 = [sample[6][0:6] for sample in gibbs_p5_inst1.MC_sample]

gibbs_p5_diag_inst1_part1 = MCMC_Diag()
gibbs_p5_diag_inst1_part1.set_mc_samples_from_list(mc_samples_part1)
gibbs_p5_diag_inst1_part1.burnin(3000)
gibbs_p5_diag_inst1_part1.show_traceplot((3,2))
gibbs_p5_diag_inst1_part1.show_hist((3,2))
gibbs_p5_diag_inst1_part1.show_acf(30, (3,2))
print("mean: :", gibbs_p5_diag_inst1_part1.get_sample_mean())
print("var   :", gibbs_p5_diag_inst1_part1.get_sample_var())
print("median:", gibbs_p5_diag_inst1_part1.get_sample_quantile([0.5]))
print("0.95ci:", gibbs_p5_diag_inst1_part1.get_sample_quantile([0.025, 0.975]))


gibbs_p5_diag_inst1_part2 = MCMC_Diag()
gibbs_p5_diag_inst1_part2.set_mc_samples_from_list(mc_samples_part2)
gibbs_p5_diag_inst1_part2.burnin(3000)
gibbs_p5_diag_inst1_part2.show_traceplot((3,2))
gibbs_p5_diag_inst1_part2.show_hist((3,2))
gibbs_p5_diag_inst1_part2.show_acf(30, (3,2))
print("mean: :", gibbs_p5_diag_inst1_part2.get_sample_mean())
print("var   :", gibbs_p5_diag_inst1_part2.get_sample_var())
print("median:", gibbs_p5_diag_inst1_part2.get_sample_quantile([0.5]))
print("0.95ci:", gibbs_p5_diag_inst1_part2.get_sample_quantile([0.025, 0.975]))


gibbs_p5_diag_inst1_part3 = MCMC_Diag()
gibbs_p5_diag_inst1_part3.set_mc_samples_from_list(mc_samples_part3)
gibbs_p5_diag_inst1_part3.burnin(3000)
gibbs_p5_diag_inst1_part3.show_traceplot((3,2))
gibbs_p5_diag_inst1_part3.show_hist((3,2))
gibbs_p5_diag_inst1_part3.show_acf(30, (3,2))
print("mean: :", gibbs_p5_diag_inst1_part3.get_sample_mean())
print("var   :", gibbs_p5_diag_inst1_part3.get_sample_var())
print("median:", gibbs_p5_diag_inst1_part3.get_sample_quantile([0.5]))
print("0.95ci:", gibbs_p5_diag_inst1_part3.get_sample_quantile([0.025, 0.975]))

