import numpy as np
import cvxopt
import matplotlib.pyplot as plt

# T is the length of time interval
# t[k] is the discrete time 

T = 200
A = np.matrix('1 T T^2/2; 0 1 T; 0 0 1')
B = np.matrix('T^3/6 T^2/2 T')

x_hat = 
# vector of position, speed and acceleration on x
y_hat = 
# vector of position, speed and acceleration on x
