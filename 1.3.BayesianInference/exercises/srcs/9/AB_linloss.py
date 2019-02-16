import numpy as np
from matplotlib import pyplot as plt

N_A = 1000
N_A_79 = 10
N_A_49 = 46
N_A_25 = 80
N_A_0 = N_A - (N_A_79 + N_A_49 + N_A_49)
observations_A = np.array([N_A_79, N_A_49, N_A_25, N_A_0])

N_B = 2000
N_B_79 = 45
N_B_49 = 84
N_B_25 = 200
N_B_0 = N_B - (N_B_79 + N_B_49 + N_B_49)
observations_B = np.array([N_B_79, N_B_49, N_B_25, N_B_0])

prior_parameters = np.array([1,1,1,1])
posterior_samples_A = np.random.dirichlet(prior_parameters + observations_A, size = 10000)
posterior_samples_B = np.random.dirichlet(prior_parameters + observations_B, size = 10000)

def expected_revenue(P):
    return 79*P[:,0] + 49*P[:,1] + 25*P[:,2] + 0*P[:,3]

posterior_expected_revenue_A = expected_revenue(posterior_samples_A)
posterior_expected_revenue_B = expected_revenue(posterior_samples_B)

plt.hist(posterior_expected_revenue_A, histtype = 'stepfilled',
         label='expected revenue of A', bins=50)
plt.hist(posterior_expected_revenue_B, histtype = 'stepfilled',
         label='expected revenue of B', bins=50, alpha = 0.8)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title("Posterior distribution of the expected revenue between pages $A$ and $B$")
plt.legend()
plt.show()

p = (posterior_expected_revenue_B > posterior_expected_revenue_A).mean()
print("Probability that page B has a higher revenue than page A: %.3f" % p)

posterior_diff = posterior_expected_revenue_B - posterior_expected_revenue_A
plt.hist(posterior_diff, histtype = 'stepfilled', color = '#7A68A6',
         label = 'difference in revenue between B and A', bins = 50)
plt.vlines(0, 0, 700, linestyles='--')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title("Posterior distribution of the delta between expected revenues of pages $A$ and $B$")
plt.legend()
plt.show()
