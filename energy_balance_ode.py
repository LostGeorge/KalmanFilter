import numpy as np

sigma = 5.670374E-8 # Stefan-Boltzmann constant
S = 1366 # Solar Constant (W/m^2 received at tip of Earth's atmosphere)
Q = S/4 # Divide by 4 for reasons (surface area)
eps = 0.41 # greenhouse factor, permittivity of atmosphere
C = 3.24E27 # Earth's planetary heat capacity

def albedo(temp):
    '''
    Gives the albedo based on temperature
    '''
    sigmoid_exp = np.exp((temp - 285) / 10)
    return 0.7 - 0.4 * sigmoid_exp / (1 + sigmoid_exp)

def temp_deriv(temp):
    return ((1 - albedo(temp)) * Q - eps * sigma * temp**4) / C

def e_balance_transition_matrix(temp, step=8E25):
    temp = temp.item()
    k = np.zeros(4)
    k[0] = step * temp_deriv(temp)
    k[1] = step * temp_deriv(temp + k[0]/2)
    k[2] = step * temp_deriv(temp + k[1]/2)
    k[3] = step * temp_deriv(temp + k[2])
    new_temp = temp + (k[0] + 2*k[1] + 2*k[2] + k[3]) / 6
    return np.array([[new_temp/temp]])
