import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import norm

# T is the length of time interval
# t is the discrete time of each step

def nextstep(T, t, x, x1, x2, x3 ,y, y1, y2, y3 ):
    k = T/t; # number of steps
   
    A = np.matrix('1 T T^2/2; 0 1 T; 0 0 1')
    B = np.matrix('T^3/6 T^2/2 T')

    X = np.empty(shape=[3,k])
    Y = np.empty(shape=[3,k])
    #put all vectors in a matrix

    X[0] = np.matrix([x, x1, x2]).T
    # position info on x
    Y[0] = np.matrix([x, x1, x2]).T
    # position info on y

    for i in range (k):
        X[:,i+1] = A * X[:,i] + B * x3
        Y[:,i+1] = A * Y[:,i] + B * y3
        print('Future foot step in x direction: \n ' + X, end=' ')
        print('Future foot step in y direction: \n ' + Y)
