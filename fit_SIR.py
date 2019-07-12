'''
Fits SIR model to given data
'''

# Module import
import SIR_funcs

import pandas as pd
import numpy as np
import scipy as sci
from scipy import stats as st
import matplotlib.pyplot as plt

# Test code to fit

# Produce synthetic data
data_beta = 0.20
data_gamma = 0.14
data_param = [data_beta, data_gamma]

data = SIR_funcs.SIR_model(data_param)
print('data beginning = ' + str(data))
I_data = data.y[1]


print('I_data' + str(I_data))
plt.figure()
plt.plot(data.t, data.y[0])

plt.figure()
plt.plot(data.t, I_data)

plt.figure()
plt.plot(data.t, data.y[2])


# Produce synthetic SEIR data
pkappa = 1
SEIR_param = [data_beta, 2, data_gamma]
SEIR_data = SIR_funcs.SEIR_model(SEIR_param)

plt.figure()
plt.plot(data.t, data.y[0], '--')
plt.plot(SEIR_data.t, SEIR_data.y[0])

plt.figure()
plt.plot(SEIR_data.t, SEIR_data.y[1])
plt.ylabel('E')

plt.figure()
plt.plot(SEIR_data.t, SEIR_data.y[2], label = 'SEIR')
plt.plot(data.t, I_data, '--', label = 'SIR')
plt.legend()
plt.ylabel('I')

plt.figure()
plt.plot(SEIR_data.t, SEIR_data.y[3])
plt.plot(data.t, data.y[2])
plt.ylabel('R')
plt.show()


'''
MCMC Set-up
'''
# Starting guess [beta, gamma]

#init_param = [st.norm.rvs(1, 1), st.norm.rvs(1, 1)]
init_param = [0.20, 0.14]

# Proposal widths
w = [0.05, 0.05]

# Prior functions (what is our prior belief about beta, gamma)

def prior_param_belief(min, max):
    return st.uniform(loc=min, scale=max-min)

def prior_param_belief_normal(mean, var):
    return st.norm(loc=mean, scale=var)

# Prior belief is that beta and gamma are centred on 1 with sd of 1
beta_prior_fun = prior_param_belief_normal(1.0, 0.9)
gamma_prior_fun = prior_param_belief_normal(5.0, 3.0)

prior_funcs = [beta_prior_fun, gamma_prior_fun]

# Number of iterates
n_iterates = 100

# Calculate the log likelihood of the initial guess
init_ll = SIR_funcs.SIR_ll(I_data, init_param) # TODO: Should I be inputting init_param or data_param?

# And log likelihood of all subsequent guesses
def run_chain(data, n_iterates, w, init_param, init_ll, prior_funcs):
    param = init_param.copy()
    ll = init_ll.copy()

    # Establish data storage space for chain
    # Where first column is ll and 1: are the n-parameters [ll, beta, gamma]
    chain = np.zeros((n_iterates, len(param) + 1))
    chain[0, 0] = ll
    chain[0, 1:] = param

    # Run MCMC
    for i in range(n_iterates):

        # Print status every 10 iterations
        if i % 10 == 0:
            print('Iteration ' + str(i) + ' of ' + str(n_iterates))

        # Gibbs loop over number of parameters (j = 0 is beta, j = 1 is gamma)
        for j in range(len(param)):

            # Propose a parameter value within prev. set widths
            prop_param = param.copy()

            # Take a random step size from a uniform distribution (that has width of w)
            prop_param[j] = prop_param[j] - (sci.stats.uniform.rvs(loc=-w[j] / 2, scale=w[j], size=1))

            # Deal with invalid proposals by leaving ll, param unchanged
            if prop_param[j] <= 0: # Invalid, so try next parameter proposal
                prop_ll = -1 * np.inf

            else:
                # Calculate LL of proposal

                prop_ll = SIR_funcs.SIR_ll(data, prop_param)

            # Decide on accept/reject
            prior_fun = prior_funcs[j]  # Grab correct prior function

            # Likelihood ratio st.norm.rvs(1, 1)
            r = np.exp(prop_ll - ll) * prior_fun.pdf(prop_param[j]) / prior_fun.pdf(param[j])
            # Is likelihood ratio less than or equal to one
            alpha = min(1, r)

            # Random number between 0 to 1
            # So will have weighted chance of possibly accepting depending on how likely the new parameter is
            test = np.random.uniform(0, 1)
            # Maybe accept
            if (test < alpha):
                ll = prop_ll.copy()
                param = prop_param.copy()
            # "Else" reject, though nothing to write

            # Store iterate
            chain[i, 0] = ll
            chain[i, 1:] = param

    return chain

print('np.shape(I_data) = ' + str(np.shape(I_data)))
print('n_iterates = ' +str(np.shape(n_iterates)))
print('w = ' + str(np.shape('w')))
print('Init_param shape = ' + str(np.shape(init_param)))
print('init_ll shape = ' + str(np.shape(init_ll)))
print('prior_funcs shape = ' + str(np.shape(prior_funcs)))
chain = run_chain(I_data, n_iterates, w, init_param, init_ll, prior_funcs)

print (chain)

# Show graphs
chain = pd.DataFrame(chain, columns=['ll', 'beta', 'gamma']) # Change np to panda array

n = np.arange(int(n_iterates)) # Creates array of iterates

chain.plot(kind= 'line', y = 'll')
plt.ylabel('Log Likelihood')
plt.xlabel('Iterate')

chain.plot(kind = 'line', y = 'beta')
#plt.plot(y = pbeta, color = 'r') # Plot true value as a single line
plt.ylabel('Estimate for beta')
plt.xlabel('Iterate')

chain.plot(kind = 'line', y = 'gamma', color = 'b')
#plt.plot(y = pgamma, color = 'r') # Plot true value as a single line
plt.ylabel('Estimate for gamma')
plt.xlabel('Iterate')

chain[['beta']].plot(kind = 'hist')

chain[['gamma']].plot(kind = 'hist')


plt.show()
