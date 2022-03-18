from bayesian_tools.MCMC_MH_Core import MCMC_Diag


#problem 1d,1e

gibbs_p5_diag_inst1_part1 = MCMC_Diag()
gibbs_p5_diag_inst1_part1.set_mc_sample_from_csv("part1")
gibbs_p5_diag_inst1_part1.burnin(10000)
gibbs_p5_diag_inst1_part1.thinning(30)
gibbs_p5_diag_inst1_part1.set_variable_names(["beta_1", "beta_2", "sigma2", "beta_0_bar", "tau2"])
# gibbs_p5_diag_inst1_part1.show_traceplot((2,3))
# gibbs_p5_diag_inst1_part1.show_hist((2,3))
# gibbs_p5_diag_inst1_part1.show_acf(40, (2,3))
# gibbs_p5_diag_inst1_part1.print_summaries(round=5)
# print("ESS - beta1", gibbs_p5_diag_inst1_part1.effective_sample_size(0)) #beta1
# print("ESS - beta2", gibbs_p5_diag_inst1_part1.effective_sample_size(1)) #beta2
# print("ESS - sigma2", gibbs_p5_diag_inst1_part1.effective_sample_size(2)) #sigma2
# print("ESS - beta_0_bar", gibbs_p5_diag_inst1_part1.effective_sample_size(3)) #beta_0_bar
# print("ESS - tau2", gibbs_p5_diag_inst1_part1.effective_sample_size(4)) #tau2


gibbs_p5_diag_inst1_part2 = MCMC_Diag()
gibbs_p5_diag_inst1_part2.set_mc_sample_from_csv("part2")
gibbs_p5_diag_inst1_part2.burnin(10000)
gibbs_p5_diag_inst1_part2.thinning(40)
# gibbs_p5_diag_inst1_part2.set_variable_names(["beta_0"+str(i) for i in range(1,31)])
# gibbs_p5_diag_inst1_part2.show_traceplot((2,3), choose_dims=[0,1,2,3,4,5])
# gibbs_p5_diag_inst1_part2.show_hist((2,3), choose_dims=[0,1,2,3,4,5])
# gibbs_p5_diag_inst1_part2.show_acf(30, (2,3), choose_dims=[0,1,2,3,4,5])
# gibbs_p5_diag_inst1_part2.print_summaries(round=5)
# print("ESS - beta_01", gibbs_p5_diag_inst1_part2.effective_sample_size(0))
# print("ESS - beta_02", gibbs_p5_diag_inst1_part2.effective_sample_size(1))
# print("ESS - beta_03", gibbs_p5_diag_inst1_part2.effective_sample_size(2))
# print("ESS - beta_04", gibbs_p5_diag_inst1_part2.effective_sample_size(3))
# print("ESS - beta_05", gibbs_p5_diag_inst1_part2.effective_sample_size(4)) 
# print("ESS - beta_06", gibbs_p5_diag_inst1_part2.effective_sample_size(5)) 



#problem 1f
#post-predictive

from random import normalvariate, seed
from statistics import mean, median, variance
from math import exp

import matplotlib.pyplot as plt
import numpy as np

seed(20220317)

def posterior_predictive_sampler(patient_idx, obs_t, sample_vec_part1, sample_vec_part2):
    #patient_idx: 1 - 30

    #part 1
    #  0       1       2       3           4
    # [beta_1, beta_2, sigma2, beta_0_bar, tau2]
    #part 2
    # beta_0i_vec

    beta_0i = sample_vec_part2[patient_idx-1]
    beta_1 = sample_vec_part1[0]
    beta_2 = sample_vec_part1[1]
    sigma2 = sample_vec_part1[2]

    normal_mean = beta_0i / (1 + beta_1 * exp(beta_2 * obs_t))
    normal_sd = sigma2 ** 0.5
    return normalvariate(normal_mean, normal_sd)

obs_t_for_patient1 = [12.07580053, 16.45865433, 20.76492589, 24.53565504, 28.90688076, 32.69701494, 36.96996798]
obs_y_for_patient1 = [10.1272364, 14.80821655, 17.49366352, 19.7437975, 23.36854628, 24.00171721, 24.8546289]

posterior_predictive_samples_for_each_t = []
for t in obs_t_for_patient1:
    posterior_predictive_samples_for_each_t.append([])
    for part1, part2 in zip(gibbs_p5_diag_inst1_part1.MC_sample, gibbs_p5_diag_inst1_part2.MC_sample):
        posterior_predictive_samples_for_each_t[-1].append(posterior_predictive_sampler(1, t, part1, part2))

grid_row = 2
grid_column= 4
plt.figure(figsize=(5*grid_column, 3*grid_row))
for i, pred_samples in enumerate(posterior_predictive_samples_for_each_t):
    plt.subplot(grid_row, grid_column, i+1)
    plt.hist(pred_samples, bins=100, density=True)
    plt.xlabel("t="+str(obs_t_for_patient1[i]))
    plt.axvline(obs_y_for_patient1[i], color="orange", linestyle="solid", linewidth=1.5)
    plt.axvline(mean(pred_samples), color="red", linestyle="solid", linewidth=0.8)
    plt.axvline(median(pred_samples), color="red", linestyle="dashed", linewidth=0.8)
    quantile_0_95 = np.quantile(pred_samples, [0.025, 0.975])
    x_axis_pts = np.linspace(quantile_0_95[0], quantile_0_95[1], num=100)
    y_axis_pts = np.zeros(len(x_axis_pts)) + 0.001
    plt.scatter(x_axis_pts, y_axis_pts, color="red", s=10, zorder=2)
plt.show()

print("Woman 1")
print("obs_t \t obs_y \t mean \t var \t 95%CI")
for i, pred_samples in enumerate(posterior_predictive_samples_for_each_t):
    cal_mean = round(mean(pred_samples),2)
    cal_var = round(variance(pred_samples),3)
    quantile_0_95 = [round(x,4) for x in np.quantile(pred_samples, [0.025, 0.975])]
    print(round(obs_t_for_patient1[i], 2),"\t", round(obs_y_for_patient1[i], 2),"\t", cal_mean,"\t", cal_var,"\t", quantile_0_95)