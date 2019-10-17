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
    #put all position vectors in a matrix

    X[:,0] = np.matrix([x, x1, x2])
    # initial position info on x
    Y[:,0] = np.matrix([x, x1, x2]) 
    # initial position info on y

    for i in range(k):
    # constrct the A and B matrices
        A = np.empty(shape=[3,3*k])
        B = np.empty(shape=[3,k])

        if (i+1)%3 == 0 :
            A[:,i] =  [1, 0, 0]
        elif (i+1)%3 == 1 :
            A[:,i] = [t_list[:,i], 1, 0]
        else:
            A[:,i] = [t_list[:,i]**2/2, t_list[:,i], 1]

        
        B[:,i] = [t_list[:,i]**3/6, t_list[:,i]**2/2, t_list[:,i]]

        X[:,i+1] = (np.stack((A[:,i], A[:,i], A[:,i]), axis = -1) * X[:,i].reshape(-1,1) + B[:,i] * x3).reshape(1,-1)
        Y[:,i+1] = (A[i] * Y[:,i].reshape(-1,1) + B[:,i] * y3).reshape(1,-1)

        

    print('Future foot step in x direction: \n ' + X, end=' ')
    print('Future foot step in y direction: \n ' + Y)

nextstep( T=100, t=10, x=10, x1=10, x2=10, x3=10, y=10, y1=10, y2=10, y3=10)
