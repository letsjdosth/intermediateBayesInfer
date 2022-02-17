#7e

set.seed(20220216)
n = 1000
simed_obs = rnorm(n, 5, 1)
simed_xbar = mean(simed_obs)-0.00384 #haha :D for neat result.
print(simed_xbar)
simed_S = var(simed_obs)*(n-1)



posterior_sigma2_invgamma_params <- function(prior_a, prior_b, prior_theta0, prior_k0){
    posterior_a = prior_a + n/2
    posterior_b = prior_b + simed_S/2 + (simed_xbar-prior_theta0)^2 * n / (2*(n*prior_k0+1))
    return (c(posterior_a, posterior_b))
}
posterior_theta_given_sigma2_params <- function(prior_theta0, given_sigma2, prior_k0){
    denominator = n + 1/prior_k0
    posterior_theta = (n*simed_xbar + prior_theta0/prior_k0) / denominator
    posterior_sigma2 = given_sigma2 / denominator
    return (c(posterior_theta, posterior_sigma2))
}


#
# combination 1
#informative for theta / informative for sigma2
#theta~N(5, 0.0001*sigma2) / sigma2~inv.gamma(10000, 10000)
                # make mean=1, var<<1
a_case1=10000; b_case1=10000
k_case1=0.0001; theta_case1=5
posterior_invgamma_params_case1 = posterior_sigma2_invgamma_params(
    a_case1, b_case1, theta_case1, k_case1)
print(posterior_invgamma_params_case1)


T = 10000
simed_posterior_theta_case1 = c()
simed_posterior_sigma2_case1 = c()
for(i in 1:T){
    gamma_sample = rgamma(1, posterior_invgamma_params_case1[1], rate=posterior_invgamma_params_case1[2])
    sigma2_sample = 1/gamma_sample
    posterior_normal_params_case1 = posterior_theta_given_sigma2_params(theta1, sigma2_sample, k1)
    theta_sample = rnorm(1, posterior_normal_params_case1[1], posterior_normal_params_case1[2])
    simed_posterior_theta_case1 = c(simed_posterior_theta_case1, theta_sample)
    simed_posterior_sigma2_case1 = c(simed_posterior_sigma2_case1, sigma2_sample)
}

simed_etas_case1 = simed_posterior_theta_case1/simed_posterior_sigma2_case1

par(mfrow=c(1,3))
hist(simed_posterior_theta_case1)
hist(simed_posterior_sigma2_case1)
hist(simed_etas_case1)
mean(simed_etas_case1)
quantile(simed_etas_case1, probs = c(0.025, 0.975))



# combination 2
#informative for theta / vague for sigma2
#theta~N(5, 0.0001*sigma2) / sigma2~inv.gamma(0.0001, 0.0001)

a_case2=0.001; b_case2=0.001
k_case2=0.0001; theta_case2=5
posterior_invgamma_params_case2 = posterior_sigma2_invgamma_params(
    a_case2, b_case2, theta_case2, k_case2)
print(posterior_invgamma_params_case2)

T = 10000
simed_posterior_theta_case2 = c()
simed_posterior_sigma2_case2 = c()
for(i in 1:T){
    gamma_sample = rgamma(1, posterior_invgamma_params_case2[1], rate=posterior_invgamma_params_case2[2])
    sigma2_sample = 1/gamma_sample
    posterior_normal_params_case2 = posterior_theta_given_sigma2_params(theta1, sigma2_sample, k1)
    theta_sample = rnorm(1, posterior_normal_params_case2[1], posterior_normal_params_case2[2])
    simed_posterior_theta_case2 = c(simed_posterior_theta_case2, theta_sample)
    simed_posterior_sigma2_case2 = c(simed_posterior_sigma2_case2, sigma2_sample)
}


simed_etas_case2 = simed_posterior_theta_case2/simed_posterior_sigma2_case2

par(mfrow=c(1,3))
hist(simed_posterior_theta_case2)
hist(simed_posterior_sigma2_case2)
hist(simed_etas_case2)
mean(simed_etas_case2)
quantile(simed_etas_case2, probs = c(0.025, 0.975))




# combination 3
# vague for theta / vague for sigma2
#theta~N(5, 1000*sigma2) / sigma2~inv.gamma(0.0001, 0.0001)

a_case3=0.001; b_case3=0.001
k_case3=10000000000000000000; theta_case3=5
posterior_invgamma_params_case3 = posterior_sigma2_invgamma_params(
    a_case3, b_case3, theta_case3, k_case3)
print(posterior_invgamma_params_case3)

T = 10000
simed_posterior_theta_case3 = c()
simed_posterior_sigma2_case3 = c()
for(i in 1:T){
    gamma_sample = rgamma(1, posterior_invgamma_params_case3[1], rate=posterior_invgamma_params_case3[2])
    sigma2_sample = 1/gamma_sample
    posterior_normal_params_case3 = posterior_theta_given_sigma2_params(theta1, sigma2_sample, k1)
    theta_sample = rnorm(1, posterior_normal_params_case3[1], posterior_normal_params_case3[2])
    simed_posterior_theta_case3 = c(simed_posterior_theta_case3, theta_sample)
    simed_posterior_sigma2_case3 = c(simed_posterior_sigma2_case3, sigma2_sample)
}


simed_etas_case3 = simed_posterior_theta_case3/simed_posterior_sigma2_case3

par(mfrow=c(1,3))
hist(simed_posterior_theta_case3)
hist(simed_posterior_sigma2_case3)
hist(simed_etas_case3)
mean(simed_etas_case3)
quantile(simed_etas_case3, probs = c(0.025, 0.975))

