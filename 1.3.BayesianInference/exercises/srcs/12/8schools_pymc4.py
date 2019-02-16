import numpy as np
import pymc4 as pm4
import tensorflow as tf # For Random Variable operation
from tensorflow_probability import edward2 as ed # For defining random variables

J = 8 # No. of schools
y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])

pymc4_non_centered_eight = pm4.Model(num_schools=J, y=y, sigma=sigma)

@pymc4_non_centered_eight.define
def process(cfg):
    mu = ed.Normal(loc=0., scale=10., name="mu")
    log_tau = ed.Normal(
        loc=5., scale=1., name="log_tau")
    theta_prime = ed.Normal(
        loc=tf.zeros(cfg.num_schools),
        scale=tf.ones(cfg.num_schools),
        name="theta_prime")
    theta = mu + tf.exp(
        log_tau) * theta_prime
    y = ed.Normal(
        loc=theta,
        scale=np.float32(cfg.sigma),
        name="y")
    return y

pymc4_non_centered_eight.observe(y=y)

pymc4_trace = pm4.sample(pymc4_non_centered_eight)

pymc4_theta = pymc4_trace['mu'][:, np.newaxis] + np.exp(pymc4_trace['log_tau'])[:, np.newaxis] * pymc4_trace['theta_prime']

print(np.mean(pymc4_theta, axis = 0))
