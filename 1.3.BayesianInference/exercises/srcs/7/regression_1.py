#Source: https://stackoverflow.com/questions/22239121/simplest-linear-model-with-pymc

import pymc as pm
import numpy as np

x_data = np.linspace(0,1,100)
y_data = np.linspace(0,1,100)

slope  = pm.Normal('slope',  mu=0, tau=10**-2)
tau    = pm.Uniform('tau', lower=0, upper=20)

@pm.deterministic
def y_gen(x=x_data, slope=slope):
  return slope * x

like = pm.Normal('likelihood', mu=y_gen, tau=tau, observed=True, value=y_data)

model = pm.Model([slope, y_gen, like, tau])
mcmc  = pm.MCMC(model)
mcmc.sample(100000, 5000)

final_guess = mcmc.trace('slope')[:].mean()

print()
print("the mean of the slope:", final_guess.mean())