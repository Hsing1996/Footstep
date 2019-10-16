import numpy as np
#import cvxopt
#import matplotlib.pyplot as plt
#import norm

def nextstep(T, t, x, x1, x2, x3 ,y, y1, y2, y3):
    # T is the length of time interval

    k = T//t; # number of steps

    t_list = np.empty(shape=[1,k]) # discrete time of each step
    for i in range(k):
        t_list[:,0] = 0
        t_list[:,i] = t_list[:,0] + i * t
    
    X = np.empty(shape=[3,k])
    Y = np.empty(shape=[3,k])
    #put all vectors in a matrix

    X[:,0] = np.matrix([x, x1, x2])
    # initial position info on x
    Y[:,0] = np.matrix([x, x1, x2]) 
    # initial position info on y

    for i in range(k):

        A = 
        B =
        A[i] = np.matrix([[1, t_list[i], t_list[i]**2/2], [0, 1, t_list[i]], [0, 0, 1]])
        B[i] = np.matrix([[t_list[i]**3/6], [t_list[i]**2/2], [t_list[i]]])

        X[:,i+1] = (A[i] * X[:,i].reshape(-1,1) + B[i] * x3).reshape(1,-1)
        Y[:,i+1] = (A[i] * Y[:,i].reshape(-1,1) + B[i] * y3).reshape(1,-1)

        

    print('Future foot step in x direction: \n ' + X, end=' ')
    print('Future foot step in y direction: \n ' + Y)

nextstep( T=100, t=10, x=10, x1=10, x2=10, x3=10, y=10, y1=10, y2=10, y3=10)
