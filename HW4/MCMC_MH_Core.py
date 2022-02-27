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

    def generate_samples(self, num_samples, pid=None, verbose=True):
        start_time = time.time()
        for i in range(num_samples):
            self.sampler()
            
            if i==100 and verbose:
                elap_time_head_iter = time.time()-start_time
                estimated_time = (num_samples/100)*elap_time_head_iter
                print("estimated running time: ", estimated_time//60, "min ", estimated_time%60, "sec")

            if i%500 == 0 and verbose and pid is not None:
                print("pid:",pid," iteration", i, "/", num_samples)
            elif i%500 == 0 and verbose and pid is None:
                print("iteration", i, "/", num_samples)
        elap_time = time.time()-start_time
        
        if pid is not None and verbose:
            print("pid:",pid, "iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
            print("acceptence rate: ", round(self.num_accept / self.num_total_iters, 4))
        elif pid is None and verbose:
            print("iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
            print("acceptence rate: ", round(self.num_accept / self.num_total_iters, 4))

        
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
    
    def set_mc_samples_from_list(self, mc_sample):
        self.MC_sample = mc_sample
        self.num_dim = len(mc_sample[0])
    
    def set_mc_sample_from_MCMC_MH(self, inst_MCMC_MH):
        self.MC_sample = inst_MCMC_MH.MC_sample
        self.num_dim = len(inst_MCMC_MH.MC_sample[0])

    def set_mc_sample_from_csv(self, file_name):
        with open(file_name + '.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for csv_row in reader:
                csv_row = [float(elem) for elem in csv_row]
                self.MC_sample(csv_row)
        self.num_dim = len(self.MC_sample[0])

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
    
    def get_specific_dim_samples(self, dim_idx):
        if dim_idx >= self.num_dim:
            raise ValueError("dimension index should be lower than number of dimension. note that index starts at 0")
        return [smpl[dim_idx] for smpl in self.MC_sample]
    
    def get_sample_mean(self):
        mean_vec = []
        for i in range(self.num_dim):
            ith_dim_samples = self.get_specific_dim_samples(i)
            mean_vec.append(mean(ith_dim_samples))
        return mean_vec

    def get_sample_var(self):
        var_vec = []
        for i in range(self.num_dim):
            ith_dim_samples = self.get_specific_dim_samples(i)
            var_vec.append(variance(ith_dim_samples))
        return var_vec


    def get_sample_quantile(self, quantile_list):
        quantile_vec = []
        for i in range(self.num_dim):
            ith_dim_samples = self.get_specific_dim_samples(i)
            quantiles = [np.quantile(ith_dim_samples, q) for q in quantile_list]
            quantile_vec.append(quantiles)
        return quantile_vec

    def show_traceplot(self, figure_grid_dim, show=True):
        grid_column= figure_grid_dim[0]
        grid_row = figure_grid_dim[1]

        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(self.num_dim):
            seq = self.get_specific_dim_samples(i)
            plt.subplot(grid_row, grid_column, i+1)
            plt.plot(range(len(seq)), seq)
        if show:
            plt.show()

    def show_hist(self, figure_grid_dim, show=True):
        grid_column= figure_grid_dim[0]
        grid_row = figure_grid_dim[1]
       
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(self.num_dim):
            plt.subplot(grid_row, grid_column, i+1)
            dim_samples = self.get_specific_dim_samples(i)
            plt.ylabel(str(i)+"-th dim")
            plt.hist(dim_samples, bins=100)
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

    def show_acf(self, maxLag, figure_grid_dim, show=True):
        grid_column= figure_grid_dim[0]
        grid_row = figure_grid_dim[1]

        subplot_grid = [i for i in range(maxLag+1)]
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(self.num_dim):
            plt.subplot(grid_row, grid_column, i+1)
            acf = self.get_autocorr(i, maxLag)
            plt.ylabel(str(i)+"-th dim")
            plt.ylim([-1,1])
            plt.bar(subplot_grid, acf, width=0.3)
            plt.axhline(0, color="black", linewidth=0.8)
        if show:
            plt.show()
    
    def show_scatterplot(self, dim_idx_horizontal, dim_idx_vertical, show=True):
        x = self.get_specific_dim_samples(dim_idx_horizontal)
        y = self.get_specific_dim_samples(dim_idx_vertical)
        plt.scatter(x, y)
        if show:
            plt.show()
    
    def effective_sample_size(self, dim_idx, sum_lags=30):
        n = len(self.MC_sample)
        auto_corr = self.get_autocorr(dim_idx, sum_lags)
        ess = n / (1 + 2*sum(auto_corr))
        return ess

if __name__ == "__main__":
    pass