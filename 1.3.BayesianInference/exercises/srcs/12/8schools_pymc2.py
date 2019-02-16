import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

num_schools = 8  # number of schools
treatment_effects = np.array(
    [28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)  # treatment effects
treatment_stddevs = np.array(
    [15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)  # treatment SE

treatment_prec = 1. / np.sqrt(treatment_stddevs)

mu = pm.Normal('mu', 0., 0.01)
log_tau = pm.Normal('log_tau', 5., 1.)

@pm.deterministic
def tau(log_tau = log_tau):
    return 1. / np.sqrt(np.exp(log_tau))

theta1 = pm.Normal('theta1', mu, tau)
theta2 = pm.Normal('theta2', mu, tau)
theta3 = pm.Normal('theta3', mu, tau)
theta4 = pm.Normal('theta4', mu, tau)
theta5 = pm.Normal('theta5', mu, tau)
theta6 = pm.Normal('theta6', mu, tau)
theta7 = pm.Normal('theta7', mu, tau)
theta8 = pm.Normal('theta8', mu, tau)

y = pm.Normal('y', [theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8], treatment_prec, value = treatment_effects, observed = True)

model = pm.Model([mu, log_tau, tau, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, y])
map_ = pm.MAP(model)
map_.fit()
mcmc = pm.MCMC(model)
#mcmc.sample(iter=200000, burn=100000, thin=10)
mcmc.sample(iter=100000, burn=0, thin=1)

pm.Matplot.plot(mcmc)

theta1_samples = mcmc.trace('theta1')[:]
plt.hist(theta1_samples)
plt.title('Posterior of theta1')
plt.show()

theta2_samples = mcmc.trace('theta2')[:]
plt.hist(theta2_samples)
plt.title('Posterior of theta2')
plt.show()

theta3_samples = mcmc.trace('theta3')[:]
plt.hist(theta3_samples)
plt.title('Posterior of theta3')
plt.show()

theta4_samples = mcmc.trace('theta4')[:]
plt.hist(theta4_samples)
plt.title('Posterior of theta4')
plt.show()

theta5_samples = mcmc.trace('theta5')[:]
plt.hist(theta5_samples)
plt.title('Posterior of theta5')
plt.show()

theta6_samples = mcmc.trace('theta6')[:]
plt.hist(theta6_samples)
plt.title('Posterior of theta6')
plt.show()

theta7_samples = mcmc.trace('theta7')[:]
plt.hist(theta7_samples)
plt.title('Posterior of theta7')
plt.show()

theta8_samples = mcmc.trace('theta8')[:]
plt.hist(theta8_samples)
plt.title('Posterior of theta8')
plt.show()

print()
print()

print(mcmc.trace('theta1')[:].mean())
print(mcmc.trace('theta2')[:].mean())
print(mcmc.trace('theta3')[:].mean())
print(mcmc.trace('theta4')[:].mean())
print(mcmc.trace('theta5')[:].mean())
print(mcmc.trace('theta6')[:].mean())
print(mcmc.trace('theta7')[:].mean())
print(mcmc.trace('theta8')[:].mean())

