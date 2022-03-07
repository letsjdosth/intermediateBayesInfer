#hw3 p7 (d)

exact_95_posterior_ci <- function(obs_n, obs_x){
    lower <- qbeta(0.025, 0.5 + obs_x, 0.5 + (obs_n - obs_x))
    upper <- qbeta(0.975, 0.5 + obs_x, 0.5 + (obs_n - obs_x))
    return(c(lower, upper))
}

mc_95_posterior_ci <- function(obs_n, obs_x, num_sim){
    mc_samples <- rbeta(num_sim, 0.5 + obs_x, 0.5 + (obs_n - obs_x))
    return(quantile(mc_samples, probs=c(0.025, 0.975)))
}

laplace_93_posterior_ci <- function(obs_n, obs_x){
    mean = (obs_x - 0.5)/(obs_n - 1)
    precision = (1/(obs_x - 0.5) + 1/(obs_n - obs_x - 0.5)) * (obs_n - 1)^2
    lower <- qnorm(0.025, mean, sqrt(1/precision))
    upper <- qnorm(0.975, mean, sqrt(1/precision))
    return(c(lower, upper))
}

#case 1: n=10, x=1
exact_95_posterior_ci(10, 1)
laplace_93_posterior_ci(10, 1)
mc_95_posterior_ci(10, 1, 100000)

#case 2: n=100, x=10
exact_95_posterior_ci(100, 10)
laplace_93_posterior_ci(100, 10)
mc_95_posterior_ci(100, 10, 100000)

#case 3: n=1000, x=100
exact_95_posterior_ci(1000, 100)
laplace_93_posterior_ci(1000, 100)
mc_95_posterior_ci(1000, 100, 100000)
