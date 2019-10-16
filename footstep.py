import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import norm

# T is the length of time interval
# t is the discrete time of each step

def nextstep(T, t, x, x1, x2, x3 ,y, y1, y2, y3 ):
    k = T/t; # number of steps
    X = np.array([x, x1, x2]).T
    # position info on x
    Y = np.array([x, x1, x2]).T
    # position info on y

    A = np.matrix('1 T T^2/2; 0 1 T; 0 0 1')
    B = np.matrix('T^3/6 T^2/2 T')

    for i in range(k):
        X = X
        Y = 
    