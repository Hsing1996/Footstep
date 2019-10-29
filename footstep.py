import numpy as np
 
def nextnsteps(T, N, x, x1, x2, x3, y, y1, y2, y3):
    # T: time interval; N: steps

    t_list = np.empty(shape=[1,N]) # discrete time of each step
    for i in range(N):
        t_list[:,i] = i * T
    
    X_hat = np.empty(shape=[3,N])
    Y_hat = np.empty(shape=[3,N])
    #put all position vectors in a matrix

    A = np.matrix([[1, T, T**2/2],[0, 1, T],[0, 0, 1]])     
    B = np.matrix([T**3/6, T**2/2, T]).T
    # constrct the A and B matrices
    X_hat[:,0] = np.matrix([x, x1, x2])
    # initial position info on x
    Y_hat[:,0] = np.matrix([x, x1, x2])
    # initial position info on y

    for k in range(N-1):
        X_hat[:, k + 1] = (A * (np.asmatrix(X_hat[:,k]).T) + B * x3).T
        Y_hat[:, k + 1] = (A * (np.asmatrix(Y_hat[:,k]).T) + B * x3).T

    print('Future foot step in x direction: \n ')
    print(X_hat)
    print(Y_hat)
    # test: print('Future foot step in y direction: \n ' + Y)

    X = np.empty(shape=[N,1])
    # X is the matrix of all x
    X1 = np.empty(shape=[N,1])
    # X1 is the matrix of all x1, aka velocity

    P_ps = np.empty(shape=[N,3])
    P_pu = np.eye(N)
    for j in range (N):
        P_ps[j] = np.array([1, j*T, (j**2 * T**2)/2])
        b = np.empty(shape=[N,1])

        b[j] = (1/6 + j/2 + j**2/2) * T**3
        # entries of P_pu
        
        for l in range(N):
           if j < l:
                P_pu[j,l] = 0
           elif j == l:
                P_pu[j,l] = 1
           else:
                for m in range(N-j):
                    P_pu[j,j-m] = b[m]
        
    # formula (7)

    print (P_ps)
    print ('This is P_pu')
    print (P_pu)
    #test
nextnsteps(T=0.01, N=100, x=10, x1=10, x2=10, x3=10, y=10, y1=10, y2=10, y3=10)
