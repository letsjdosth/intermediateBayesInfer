set.seed(20220216)

simed_obs_n4 <- rnorm(50, 4, sqrt(10))
simed_obs_n1.5 <- rnorm(50, 1.5, sqrt(10))

par(mfrow=c(1, 1))
hist(simed_obs_n4)
hist(simed_obs_n1.5)

# drawing posterior samples
update_weight_normalmixture <- function(prior_weight, mu_vec, x_bar){
    tau = 1
    sigma = sqrt(10)
    n = 50
    new_weight = prior_weight * exp(-0.5*(n/(n*tau^2+sigma^2))*(mu_vec-x_bar)^2)
    new_weight = new_weight/sum(new_weight)
    return(new_weight)
}

update_normalmixture_parameters <- function(mu, x_bar){
    tau = 1
    sigma = sqrt(10)
    n = 50

    scale_param  = 1/((n/sigma^2) + (1/tau^2))
    loc_param = scale_param * (x_bar*n/sigma^2 + mu/tau^2)
    return(c(loc_param, sqrt(scale_param)))
}

prior_weight = c(1/3, 1/3, 1/3)
prior_mu_vec = c(-3,0,3)
posterior_weight_n4 = update_weight_normalmixture(
    prior_weight, prior_mu_vec, mean(simed_obs_n4))
print(round(posterior_weight_n4, 4))

posterior_weight_n1.5 = update_weight_normalmixture(
    prior_weight, prior_mu_vec, mean(simed_obs_n1.5))
print(round(posterior_weight_n1.5, 4))


N = 10000
chosen_mixture_n4 = sample(c(1,2,3), N, replace=TRUE, prob=posterior_weight_n4)
chosen_mixture_n1.5 = sample(c(1,2,3), N, replace=TRUE, prob=posterior_weight_n1.5)

chosen_mixture_count_n1.5 = c(
    sum(chosen_mixture_n1.5==1),
    sum(chosen_mixture_n1.5==2),
    sum(chosen_mixture_n1.5==3))
chosen_mixture_count_n1.5
chosen_mixture_count_n4 = c(
    sum(chosen_mixture_n4==1),
    sum(chosen_mixture_n4==2), 
    sum(chosen_mixture_n4==3))
chosen_mixture_count_n4

posterior_sample_vec_n4 = c()
for(i in 1:3){
    params = update_normalmixture_parameters(prior_mu_vec[i], mean(simed_obs_n4))
    posterior_sample_vec_n4 = c(
        posterior_sample_vec_n4, 
        rnorm(chosen_mixture_count_n4[i], params[1], params[2]))
}
hist(posterior_sample_vec_n4, breaks=100)



posterior_sample_vec_n1.5 = c()
for(i in 1:3){
    params = update_normalmixture_parameters(prior_mu_vec[i], mean(simed_obs_n1.5))
    posterior_sample_vec_n1.5 = c(
        posterior_sample_vec_n1.5, 
        rnorm(chosen_mixture_count_n1.5[i], params[1], params[2]))
}
hist(posterior_sample_vec_n1.5, breaks=100)
mean(postpredictive_sample_vec_n1.5)
var(postpredictive_sample_vec_n1.5)

#draw posterior predictive samples
#1. using posterior samples
K=10000
chosen_mu = sample(posterior_sample_vec_n4, K, replace=TRUE)
posterior_predictive_sample_n4 = c()
for(i in 1:K){
    posterior_predictive_sample_n4 = c(posterior_predictive_sample_n4, rnorm(1, chosen_mu[i], sqrt(10)))
}
hist(posterior_predictive_sample_n4, breaks=100)
mean(postpredictive_sample_vec_n4)
var(postpredictive_sample_vec_n4)



K=10000
chosen_mu = sample(posterior_sample_vec_n1.5, K, replace=TRUE)
posterior_predictive_sample_n1.5 = c()
for(i in 1:K){
    posterior_predictive_sample_n1.5 = c(posterior_predictive_sample_n1.5, rnorm(1, chosen_mu[i], sqrt(10)))
}
hist(posterior_predictive_sample_n1.5, breaks=100)
mean(postpredictive_sample_vec_n1.5)
var(postpredictive_sample_vec_n1.5)

#2. using posterior predictive distribution directly

K = 10000
chosen_mixture_n4 = sample(c(1,2,3), K, replace=TRUE, prob=posterior_weight_n4)
chosen_mixture_n1.5 = sample(c(1,2,3), K, replace=TRUE, prob=posterior_weight_n1.5)

chosen_mixture_count_n1.5 = c(
    sum(chosen_mixture_n1.5==1),
    sum(chosen_mixture_n1.5==2),
    sum(chosen_mixture_n1.5==3))
chosen_mixture_count_n1.5
chosen_mixture_count_n4 = c(
    sum(chosen_mixture_n4==1),
    sum(chosen_mixture_n4==2), 
    sum(chosen_mixture_n4==3))
chosen_mixture_count_n4

update_normalmixture_predictive_parameters <- function(mu, x_bar){
    tau = 1
    sigma = sqrt(10)
    n = 50

    scale_param  = 1/((n/sigma^2) + (1/tau^2))
    loc_param = scale_param * (x_bar*n/sigma^2 + mu/tau^2)
    return(c(loc_param, sqrt(scale_param + sigma^2)))
}

postpredictive_sample_vec_n4_v2 = c()
for(i in 1:3){
    params = update_normalmixture_predictive_parameters(
        prior_mu_vec[i], mean(simed_obs_n4))
    postpredictive_sample_vec_n4_v2 = c(
        postpredictive_sample_vec_n4_v2, 
        rnorm(chosen_mixture_count_n4[i], params[1], params[2]))
}
hist(postpredictive_sample_vec_n4_v2, breaks=100)
mean(postpredictive_sample_vec_n4_v2)
var(postpredictive_sample_vec_n4_v2)


postpredictive_sample_vec_n1.5_v2 = c()
for(i in 1:3){
    params = update_normalmixture_predictive_parameters(
        prior_mu_vec[i], mean(simed_obs_n1.5))
    postpredictive_sample_vec_n1.5_v2 = c(
        postpredictive_sample_vec_n1.5_v2, 
        rnorm(chosen_mixture_count_n1.5[i], params[1], params[2]))
}
hist(postpredictive_sample_vec_n1.5_v2, breaks=100)
mean(postpredictive_sample_vec_n1.5_v2)
var(postpredictive_sample_vec_n1.5_v2)