import numpy as np
#import cvxopt
#import matplotlib.pyplot as plt
#import norm

def nextstep(T, t, x, x1, x2, x3 ,y, y1, y2, y3):
# T is the length of time interval
# t is the discrete time of each step

    k = T/t; # number of steps
   
    A = np.matrix('1 T T^2/2; 0 1 T; 0 0 1')
    B = np.matrix('T^3/6 T^2/2 T')

    X = np.empty(shape=[3,k])
    Y = np.empty(shape=[3,k])
    #put all vectors in a matrix

    X[:,0] = np.matrix([x, x1, x2]).T
    # position info on x
    Y[:,0] = np.matrix([x, x1, x2]).T
    # position info on y

    for i in range(k):
        X[:,i+1] = A * X[:,i] + B * x3
        Y[:,i+1] = A * Y[:,i] + B * y3
        print('Future foot step in x direction: \n ' + X, end=' ')
        print('Future foot step in y direction: \n ' + Y)

nextstep( T=100, t=10, x=10, x1=10, x2=10, x3=10, y=10, y1=10, y2=10, y3=10)
