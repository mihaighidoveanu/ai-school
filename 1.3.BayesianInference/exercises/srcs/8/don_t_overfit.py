import numpy as np
import pymc as pm
from matplotlib import pyplot as plt
from sklearn import metrics


d = 200
n = 20000 # number of data points
x_data = np.random.random((n,d))
A = np.random.random(d)
isel = np.random.binomial(1, 0.5, d)
A = A * isel
y_data = np.dot(x_data, A)
y_data = y_data - y_data.mean()
y_data = 0.5 * (np.sign(y_data) + 1)

it = np.random.choice(20000, 250)

ix_training = it
ix_testing = np.ones(20000)
ix_testing[it] = 0
ix_testing = ix_testing == 1

training_data = x_data[ix_training]
testing_data = x_data[ix_testing]

training_labels = y_data[ix_training]
testing_labels = y_data[ix_testing]

to_include = pm.Bernoulli("to_include", 0.5, size=200)

coef = pm.Uniform("coefs", 0, 1, size=200)

@pm.deterministic
def Z(coef=coef, to_include=to_include, data=training_data):
    ym = np.dot(to_include * training_data, coef)
    return ym - ym.mean()

@pm.deterministic
def T(z=Z):
    return 0.45 * (np.sign(z) + 1.1)

obs = pm.Bernoulli("obs", T, value=training_labels, observed=True)

model = pm.Model([to_include, coef, Z, T, obs])
map_ = pm.MAP(model)
map_.fit()

mcmc = pm.MCMC(model)
mcmc.sample(1000000, 990000)

t_trace = mcmc.trace("T")[:]
include_trace = mcmc.trace("to_include")[:]
coef_trace = mcmc.trace("coefs")[:]

print()
print()
#print((np.round(t_trace[-500:-400, :]).mean(axis=0) == training_labels).mean())
print((np.round(t_trace.mean(axis=0)) == training_labels).mean())

#include = include_trace[-500:-400, :].mean(axis=0)
#coef = coef_trace[-500:-400, :].mean(axis=0)
include = include_trace.mean(axis=0)
coef = coef_trace.mean(axis=0)

yh = np.dot(include * training_data, coef)
yh = yh - yh.mean()
print(metrics.roc_auc_score(training_labels, yh))
yh = 0.5 * (np.sign(yh) + 1)
print((yh == training_labels).mean())

yt = np.dot(include * testing_data, coef)
yt = yt - yt.mean()
print(metrics.roc_auc_score(testing_labels, yt))
yt = 0.5 * (np.sign(yt) + 1)
print((yt == testing_labels).mean())

# histogram of the samples:

plt.bar(range(200), isel)
plt.show()

plt.bar(range(200), include)
plt.show()

plt.imshow(include_trace[-10000:, :], aspect="auto", interpolation="none")
plt.show()
