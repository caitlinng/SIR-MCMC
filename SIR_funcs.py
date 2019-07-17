import pandas as pd
import numpy as np
import scipy as sci
import scipy.stats as st
from scipy.integrate import solve_ivp
from scipy import stats as st
import matplotlib.pyplot as plt

'''
SIR_model takes given parameters [beta, gamma]
and returns solution to SIR differential equations as dictionary where t:[time], y:[S], [I], [R] at each time point (each day in 52 weeks)
'''

def SIR_model(param):  # Where param = [beta, gamma]
    # Set parameters
    beta = param[0]
    gamma = param[1]
    #beta = 0.20
    #gamma = 0.14

    if beta/gamma <= 1:
        print ('Reproductive rate too low to sustain')
    # Total population
    N = 1000

    # Define time
    time = np.linspace(0, 26*7)

    # Define initial conditions
    I0 = 1
    S0 = N - I0
    R0 = 0

    SIR_init = [S0, I0, R0]

    # Create storage for SIR parameters beta, gamma, N
#    SIR_params = pd.DataFrame({'beta': [beta], 'gamma': [gamma], 'N': [N]})

    def SIR_rhs(t, y):
        # Defining SIR model's right-hand side equations
        S, I, R = y

        Sdot = -beta * S * I / N
        Idot = (beta * S * I / N) - (gamma * I)
        Rdot = gamma * I

        SIR_dot = [Sdot, Idot, Rdot]
        return SIR_dot

    sol = solve_ivp(SIR_rhs, [time[0], time[-1]], SIR_init, method='RK45', t_eval=time)

    return sol


def SEIR_model(param):  # Where param = [beta, kappa, gamma]
    # Set parameters
    beta = param[0]
    kappa = param[1]
    gamma = param[2]

    if beta/gamma <= 1:
        print ('Reproductive rate too low to sustain')

    # Total population
    N = 1000

    # Define time
    time = np.linspace(0, 30*7)

    # Define initial conditions
    I0 = 1
    S0 = N - I0
    E0 = 0
    R0 = 0

    SEIR_init = [S0, E0, I0, R0]

    def SEIR_rhs(t, y):
        # Defining SIR model's right-hand side equations
        S, E, I, R = y

        Sdot = -beta * S * I / N
        Edot = beta * S * I / N - (kappa * E)
        Idot = (kappa * E) - (gamma * I)
        Rdot = gamma * I

        SEIR_dot = [Sdot, Edot, Idot, Rdot]
        return SEIR_dot

    sol = solve_ivp(SEIR_rhs, [time[0], time[-1]], SEIR_init, method='RK45', t_eval=time)

    return sol

def SIR_ll(I_data, param): # Where I_data = I (infected individuals) as retrieved from data
    # Obtain model values for I, given new parameters
    model_data = SIR_model(param)
    I_model = SIR_model(param).y[1]
   # print('For param = ' + str(param) +'I_model is: ' + str(I_model))
    #print('For param = ' + str(param) +'I_data is: ' + str(I_data))
    ll = 0

    for k in range(len(I_data)):
        new_ll = st.poisson.logpmf(k=int(I_data[k]), mu=int(I_model[k]))
#        new_ll = st.norm.logpdf(I_data[k], loc=int(I_model[k]), scale=int(I_model[k]))
        #    print('new_ll = ' + str(new_ll))

        ll = ll + new_ll
        return ll

    '''

    plt.figure()
    plt.plot(model_data.t, I_model, label='model')
    plt.plot(model_data.t, I_data, '--', label='data')
    plt.legend()
    plt.ylabel('I')
    plt.show()

'''
    #      if np.isnan(ll):  # can't move from -inf
#            print(str(k) + ' was a nan ')


