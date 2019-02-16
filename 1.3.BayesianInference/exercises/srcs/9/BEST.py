import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

N = 250
mu_A, std_A = 30, 4
mu_B, std_B = 26, 7

# create durations (seconds) users are on the pages for

durations_A = np.random.normal(mu_A, std_A, size = N)
durations_B = np.random.normal(mu_B, std_B, size = N)

pooled_mean = np.r_[durations_A, durations_B].mean()
pooled_std = np.r_[durations_A, durations_B].std()

tau = 1. / np.sqrt(1000. * pooled_std) # PyMC uses a precision
                                       # parameter, 1/sigma**2

mu_A = pm.Normal("mu_A", pooled_mean, tau)
mu_B = pm.Normal("mu_B", pooled_mean, tau)

std_A = pm.Uniform("std_A", pooled_std / 1000., 1000. * pooled_std)
std_B = pm.Uniform("std_B", pooled_std / 1000., 1000. * pooled_std)

nu_minus_1 = pm.Exponential("nu-1", 1./29)

obs_A = pm.NoncentralT("obs_A", mu_A, 1.0 / std_A ** 2, nu_minus_1 + 1,
                       observed = True, value = durations_A)
obs_B = pm.NoncentralT("obs_B", mu_B, 1.0 / std_B ** 2, nu_minus_1 + 1,
                       observed = True, value = durations_B)

mcmc = pm.MCMC([obs_A, obs_B, mu_A, mu_B, std_A, std_B, nu_minus_1])
mcmc.sample(25000,10000)

mu_A_trace, mu_B_trace = mcmc.trace('mu_A')[:], mcmc.trace('mu_B')[:]
std_A_trace, std_B_trace = mcmc.trace('std_A')[:], mcmc.trace('std_B')[:]
nu_trace = mcmc.trace("nu-1")[:] + 1

def _hist(data, label, **kwargs):
    return plt.hist(data, bins=40, histtype='stepfilled',
                    alpha = .95, label = label, **kwargs)

ax = plt.subplot(3,1,1)
_hist(mu_A_trace,'A')
_hist(mu_B_trace,'B')
plt.legend()
plt.title('Posterior distributions of $\mu$')

ax = plt.subplot(3,1,2)
_hist(std_A_trace, 'A')
_hist(std_B_trace, 'B')
plt.legend()
plt.title('Posterior distributions of $\sigma$')

ax = plt.subplot(3,1,3)
_hist(nu_trace,'', color='#7A68A6')
plt.title(r'Posterior distribution of $\nu$')
plt.xlabel('Value')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

