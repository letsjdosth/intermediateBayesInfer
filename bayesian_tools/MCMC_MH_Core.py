import time
import csv
from math import log
from random import seed, uniform
from statistics import mean, variance

import numpy as np
import matplotlib.pyplot as plt


class MCMC_MH:
    def __init__(self, log_target_pdf, log_proposal_pdf, proposal_sampler, initial, random_seed):
        self.log_target_pdf = log_target_pdf #arg (smpl)
        self.log_proposal_pdf = log_proposal_pdf #arg (from_smpl, to_smpl)
        self.proposal_sampler = proposal_sampler #function with argument (smpl)
        
        self.initial = initial
        
        self.MC_sample = [initial]

        self.num_total_iters = 0
        self.num_accept = 0

        self.random_seed = random_seed
        seed(random_seed)
        

    def log_r_calculator(self, candid, last):
        log_r = (self.log_target_pdf(candid) - self.log_proposal_pdf(from_smpl=last, to_smpl=candid) - \
             self.log_target_pdf(last) + self.log_proposal_pdf(from_smpl=candid, to_smpl=last))
        return log_r

    def sampler(self):
        last = self.MC_sample[-1]
        candid = self.proposal_sampler(last) #기존 state 집어넣게
        unif_sample = uniform(0, 1)
        log_r = self.log_r_calculator(candid, last)
        # print(log(unif_sample), log_r) #for debug
        if log(unif_sample) < log_r:
            self.MC_sample.append(candid)
            self.num_total_iters += 1
            self.num_accept += 1
        else:
            self.MC_sample.append(last)
            self.num_total_iters += 1

    def generate_samples(self, num_samples, pid=None, verbose=True, print_iter_cycle=500):
        start_time = time.time()
        for i in range(num_samples):
            self.sampler()
            
            if i==100 and verbose:
                elap_time_head_iter = time.time()-start_time
                estimated_time = (num_samples/100)*elap_time_head_iter
                print("estimated running time: ", estimated_time//60, "min ", estimated_time%60, "sec")

            if i%print_iter_cycle == 0 and verbose and pid is not None:
                print("pid:",pid," iteration", i, "/", num_samples)
            elif i%print_iter_cycle == 0 and verbose and pid is None:
                print("iteration", i, "/", num_samples)
        elap_time = time.time()-start_time
        
        if pid is not None and verbose:
            print("pid:",pid, "iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
            print("acceptance rate: ", round(self.num_accept / self.num_total_iters, 4))
        elif pid is None and verbose:
            print("iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
            print("acceptance rate: ", round(self.num_accept / self.num_total_iters, 4))

        
    def write_samples(self, filename: str):
        with open(filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sample in self.MC_sample:
                csv_row = sample
                writer.writerow(csv_row)



class MCMC_Diag:
    def __init__(self):
        self.MC_sample = []
        self.num_dim = None
        self.variable_names = None
    
    def set_mc_samples_from_list(self, mc_sample, variable_names=None):
        self.MC_sample = mc_sample
        self.num_dim = len(mc_sample[0])
        if variable_names is not None:
            self.set_variable_names(variable_names)

    def set_mc_sample_from_MCMC_MH(self, inst_MCMC_MH, variable_names=None):
        self.MC_sample = inst_MCMC_MH.MC_sample
        self.num_dim = len(inst_MCMC_MH.MC_sample[0])
        if variable_names is not None:
            self.set_variable_names(variable_names)

    def set_mc_sample_from_csv(self, file_name, variable_names=None):
        with open(file_name + '.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for csv_row in reader:
                csv_row = [float(elem) for elem in csv_row]
                self.MC_sample(csv_row)
        self.num_dim = len(self.MC_sample[0])
        if variable_names is not None:
            self.set_variable_names(variable_names)

    def set_variable_names(self, name_list):
        self.variable_names = name_list

    def write_samples(self, filename: str):
        with open(filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sample in self.MC_sample:
                csv_row = sample
                writer.writerow(csv_row)

    def burnin(self, num_burn_in):
        self.MC_sample = self.MC_sample[num_burn_in-1:]

    def thinning(self, lag):
        self.MC_sample = self.MC_sample[::lag]
    
    def _round_list(self, list_obj, round_digit):
        rounded = [round(x, round_digit) for x in list_obj]
        return rounded

    def get_specific_dim_samples(self, dim_idx):
        if dim_idx >= self.num_dim:
            raise ValueError("dimension index should be lower than number of dimension. note that index starts at 0")
        return [smpl[dim_idx] for smpl in self.MC_sample]
    
    def get_sample_mean(self, round=None):
        mean_vec = []
        for i in range(self.num_dim):
            ith_dim_samples = self.get_specific_dim_samples(i)
            mean_vec.append(mean(ith_dim_samples))
        if round is not None:
            mean_vec = self._round_list(mean_vec, round)
        return mean_vec


    def get_sample_var(self, round=None):
        var_vec = []
        for i in range(self.num_dim):
            ith_dim_samples = self.get_specific_dim_samples(i)
            var_vec.append(variance(ith_dim_samples))
        if round is not None:
            var_vec = self._round_list(var_vec, round)
        return var_vec

    def get_sample_quantile(self, quantile_list, round=None):
        quantile_vec = []
        for i in range(self.num_dim):
            ith_dim_samples = self.get_specific_dim_samples(i)
            quantiles = [np.quantile(ith_dim_samples, q) for q in quantile_list]
            quantile_vec.append(quantiles)
        
        if round is not None:
            quantile_vec = [self._round_list(x, round) for x in quantile_vec]
        return quantile_vec

    def print_summaries(self, round = None, name=False):
        #name/mean/var/95%CI
        mean_vec = self.get_sample_mean(round=round)
        var_vec = self.get_sample_var(round=round)
        cred95_interval_vec = self.get_sample_quantile([0.025, 0.975], round=round)


        print("param \t mean \t var \t 95%CI")
        if name:
            for var_name, mean_val, var_val, cred95_vals in zip(self.variable_names, mean_vec, var_vec, cred95_interval_vec):
                print(var_name, "\t", mean_val, "\t", var_val, "\t", cred95_vals)
        else:
            for i, (mean_val, var_val, cred95_vals) in enumerate(zip(mean_vec, var_vec, cred95_interval_vec)):
                print(i,"th", "\t", mean_val, "\t", var_val, "\t", cred95_vals)



    def show_traceplot_specific_dim(self, dim_idx, show=False, name=False):
        traceplot_data = self.get_specific_dim_samples(dim_idx)
        if name:
            plt.ylabel(self.variable_names[dim_idx])
        else:
            plt.ylabel(str(dim_idx)+"th dim")
        plt.plot(range(len(traceplot_data)), traceplot_data)
        if show:
            plt.show()

    def show_traceplot(self, figure_grid_dim, show=True, name=False):
        grid_column= figure_grid_dim[0]
        grid_row = figure_grid_dim[1]

        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(self.num_dim):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_traceplot_specific_dim(i, name = name)
        if show:
            plt.show()

    
    def show_hist_specific_dim(self, dim_idx, show=False, name=False):
        hist_data = self.get_specific_dim_samples(dim_idx)
        if name:
            plt.ylabel(self.variable_names[dim_idx])
        else:
            plt.ylabel(str(dim_idx)+"th dim")

        plt.hist(hist_data, bins=100)
        if show:
            plt.show()

    def show_hist(self, figure_grid_dim, show=True, name=False):
        grid_column= figure_grid_dim[0]
        grid_row = figure_grid_dim[1]
       
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(self.num_dim):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_hist_specific_dim(i, name = name)
        if show:
            plt.show()

    def get_autocorr(self, dim_idx, maxLag):
        y = self.get_specific_dim_samples(dim_idx)
        acf = []
        y_mean = mean(y)
        y = [elem - y_mean  for elem in y]
        n_var = sum([elem**2 for elem in y])
        for k in range(maxLag+1):
            N = len(y)-k
            n_cov_term = 0
            for i in range(N):
                n_cov_term += y[i]*y[i+k]
            acf.append(n_cov_term / n_var)
        return acf
    
    def show_acf_specific_dim(self, dim_idx, maxLag, show=False, name=False):
        grid = [i for i in range(maxLag+1)]
        acf = self.get_autocorr(dim_idx, maxLag)
        plt.ylim([-1,1])
        
        if name:
            plt.ylabel(self.variable_names[dim_idx])
        else:
            plt.ylabel(str(dim_idx)+"th dim")

        plt.bar(grid, acf, width=0.3)
        plt.axhline(0, color="black", linewidth=0.8)
        if show:
            plt.show()

    def show_acf(self, maxLag, figure_grid_dim, show=True, name=False):
        grid_column= figure_grid_dim[0]
        grid_row = figure_grid_dim[1]

        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(self.num_dim):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_acf_specific_dim(i, maxLag, name = name)
        if show:
            plt.show()
    
    def show_scatterplot(self, dim_idx_horizontal, dim_idx_vertical, show=True, name=False):
        x = self.get_specific_dim_samples(dim_idx_horizontal)
        y = self.get_specific_dim_samples(dim_idx_vertical)
        plt.scatter(x, y)
        if name:
            plt.xlabel(self.variable_names[dim_idx_horizontal])
            plt.ylabel(self.variable_names[dim_idx_vertical])
        else:
            plt.xlabel(str(dim_idx_horizontal)+"th dim")
            plt.ylabel(str(dim_idx_vertical)+"th dim")
        if show:
            plt.show()
    
    def effective_sample_size(self, dim_idx, sum_lags=30):
        n = len(self.MC_sample)
        auto_corr = self.get_autocorr(dim_idx, sum_lags)
        ess = n / (1 + 2*sum(auto_corr))
        return ess

    def DIC(self, log_likelihood_func_given_data, pt_estimate_method = "mean"): 
        # Caution: need to test. this function is not tested

        pt_est = None
        if pt_estimate_method is "mean":
            pt_est = self.get_sample_mean()
        else:
            raise ValueError("only mean pt_estimate_method is implemented now :D")
        
        def deviance_D(param_vec):
            return log_likelihood_func_given_data(param_vec) * (-2)
        deviance_at_pt_est = deviance_D(pt_est)
        deviances_at_all_samples = [deviance_D(x) for x in self.MC_sample]
        expected_deviance = mean(deviances_at_all_samples)
        return expected_deviance * 2 - deviance_at_pt_est



# histogram with mean/median/user_setted/95%CI
# traceplot with mean/median/user_setted


if __name__ == "__main__":
    pass