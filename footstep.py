import numpy as np
import cvxopt
import plotly


def nextmsteps(T, N, x, x1, x2, y, y1, y2, h, g, X_fc, Y_fc, alpha, beta, gamma, m):
    # # T: time interval; N: time steps; m: prediction horizon;
    # X_fc = 1*1
    # Y_fc = 1*1
    # # current foot position
    

    # discrete time of each step
    t_list = np.empty(shape=[N, 1]) 
    for i in range(N):
        t_list[i] = i * T
    print(t_list)

    # put all position vectors in a matrix
    X_hat = np.zeros((3,N))
    Y_hat = np.zeros((3,N))
    
    # constrct the A and B matrices
    A = np.matrix([[1, T, T**2/2],[0, 1, T],[0, 0, 1]])     
    B = np.matrix([T**3/6, T**2/2, T]).T

    # initialize X3 Y3
    X3 = np.zeros((N,1))
    Y3 = np.zeros((N,1))
    
    # initial position info (put in columns
    X_hat[:,0] = np.matrix([x, x1, x2])
    Y_hat[:,0] = np.matrix([y, y1, y2])
    
    for k in range(N-1):
        X_hat[:, k + 1] = (A * (np.asmatrix(X_hat[:,k]).T) + B * X3[k]).T
        Y_hat[:, k + 1] = (A * (np.asmatrix(Y_hat[:,k]).T) + B * Y3[k]).T

    # test: print('Future foot step in y direction: \n ' + Y)
    print('Future foot step in x direction: \n ')
    print(X_hat)
    print(Y_hat)

    # X is the matrix of all x, aka position
    X = np.zeros((N,1))
    Y = np.zeros((N,1))
    
    # construct P_ps, P_pu
    P_ps = np.zeros((N,3))
    P_pu = np.zeros((N,N))

    # b:entries of P_pu
    b = np.zeros((N,1))

    for j in range (N):
        P_ps[j] = np.array([1, j*T, (j**2 * T**2)/2])
    
        b[j] = (1/6 + j/2 + (j**2)/2) * (T**3)
        
        for l in range(N):
            if j < l:
                P_pu[j,l] = 0
            else:
                P_pu[j,j-l] = b[l]
    # formula (7)

    # positions
    for i in range(N-1):
        X[i] = np.dot(P_ps[i], X_hat[:,i]) + np.dot(P_pu[i], X3)
        Y[i] = np.dot(P_ps[i], Y_hat[:,i]) + np.dot(P_pu[i], Y3)

    #test
    print ('This is P_ps')
    print (P_ps)
    print ('This is P_pu')
    print (P_pu)
    print ('This is b' )
    print (b)

    
    # construct P_vs, P_vu
    P_vs = np.zeros((N,3))
    P_vu = np.eye(N)
    # entries of P_vu
    c = np.zeros((N,1))

    for j in range (N-1):
        P_vs[j] = np.array([0, 1, j*T])
        
        c[j] = (1/2 + j) * (T**2)
        
        for l in range(N-1):
            if j < l:
                P_vu[j,l] = 0
            else:
                P_vu[j,j-l] = c[l]
    # formula (8)

    # x,y velocity (put in column)
    X1 = np.zeros((N,1))
    Y1 = np.zeros((N,1))
    for i in range(N-1):
        X1[i] = np.dot(P_vs[i], X_hat[:,i]) + np.dot(P_vu[i], X3)
        Y1[i] = np.dot(P_vs[i], X_hat[:,i]) + np.dot(P_vu[i], Y3)
    
    #test
    print (P_vs)
    print ('This is P_vu')
    print (P_vu)
    print ('This is c')
    print (c)
    
    # z_x = np.zeros((1*N))
    # z_y = np.zeros((1*N))
    # z_x[k] = np.dot([1, 0, -h/g], X_hat[:,k])
    # z_y[k] = np.dot([1, 0, -h/g], Y_hat[:,k])
    
    # #test
    # print(z_x)
    # print(z_y)
    
    Z_x = np.zeros((N*1))
    Z_y = np.zeros((N*1))

    P_zs = np.zeros((N,3))
    P_zu = np.eye(N)

    # d: entries of P_zu
    d = np.empty(shape=[N,1])

    for j in range (N-1):
        P_zs[j] = np.array([1, j*T, (j**2 * T**2)/2 - h/g])
        
        d[j] = (T**3)/6 + (j*T**3)/2 + ((j**2)*(T**3))/2 - (h*T)/g
        
        for l in range(N-1):
            if j < l:
                P_zu[j,l] = 0
            else:
                P_zu[j,j-l] = d[l]    
    # formula (9)

    for i in range(N-1):
        Z_x[i] = np.dot(P_zs[i], X_hat[:,i]) + np.dot(P_zu[i], X3)
        Z_y[i] = np.dot(P_zs[i], Y_hat[:,i]) + np.dot(P_zu[i], Y3)
    
    # test
    print(Z_x)
    print(Z_y)
    
    # # Z_xref = np.empty(shape = [N*1])
    # # Z_yref = np.empty(shape = [N*1])
    # # Z_xref[i] = i*T
    # # Z_yref[i] = i*T
    # # reference position in original is fixed in advance

    # #QP begins
    # u = np.hstack(X3, Y3).T
    # # uk in formula (13), u shape = 2N*1

    # Q_prime = gamma * np.dot(P_zu.T, P_zu) + alpha * np.eye(N)
    # # formula (15)
    # Q = np.kron(np.eye(2),Q_prime)
    # # formula (14)， Q shape = 2N*2N

    X_f = np.zeros((3*m))
    Y_f = np.zeros((3*m))
    # foot position in the next m steps 


    Z_xref = np.zeros((N*N))
    Z_yref = np.zeros((N*N))
    # formula (20)(21)

    n = N/(m+1)  # n: times steps of each step
    n = int(n)

    # set reference velocity
    X1_ref = np.zeros((N,1))
    Y1_ref = np.zeros((N,1))
    for i in range(N):
        X1_ref[i] = 0.3
        Y1_ref[i] = 0.3

    for k in range(n*m+1): # k is in 0~15
        
        #initialize U_c, U
        U_c = np.zeros((N, 1))
        U = np.ones((N, m))

        # construct U_c
        U_c[k:(k+n)] = np.ones(n).reshape(-1,1)

        if k in range(n):
            #construct U_c column 1
            U[0:(n+k), 0] = np.zeros(n+k)
            U[(k+n*m):(n*(m+1)), 0] = np.zeros(n*(m+1)-(k+n*m))

            # construct U_c column 2
            U[k:(k+2*n), 1] = np.zeros(2*n)

        if k in range(n,2*n+1):
            #construct U_c column 1
            U[(k-n):(k+n), 0] = np.zeros(2*n)

            # construct U_c column 2
            U[0:(k-n), 1] = np.zeros(k-n)
            U[k:(n*(m+1)), 1] = np.zeros((m+1)*n - k)
        
        #test
        print(U_c)
        print(U)
        # formula (22)
    
        Q_kprime11 = alpha * np.eye(N) + beta * np.dot(P_vu.T, P_vu) + gamma * np.dot(P_zu.T, P_zu)
        Q_kprime12 = - gamma *np.dot(P_zu.T, U)
        Q_kprime21 = - gamma *np.dot(U.T, P_zu)
        Q_kprime22 = - gamma *np.dot(U.T, U)
        Q_kprime1 = np.hstack((Q_kprime11, Q_kprime12))
        Q_kprime2 = np.hstack((Q_kprime21, Q_kprime22))
        Q_kprime = np.vstack((Q_kprime1, Q_kprime2))
        # formula (24)

        Q_k = np.kron(np.eye(2),Q_kprime)  # formula (23)

        
        # formula (25)
        Pk1 = beta * P_vu.T.dot(P_vs.dot(X_hat[:,k])-X1_ref) + gamma * P_zu.T.dot(P_zs.dot(X_hat[:,k]) - np.dot(U_c, X_fc))
        Pk1 = Pk1.reshape(-1, 1)

        Pk2 = - gamma * np.dot((U.T),(np.dot(P_zs, X_hat[:,k]) - np.dot(P_zs, X_hat[:,k])))
        Pk2 = Pk2.reshape(-1, 1)

        Pk3 = beta * P_vu.T.dot(P_vs.dot(Y_hat[:,k])-Y1_ref) + gamma * P_zu.T.dot(P_zs.dot(Y_hat[:,k]) - np.dot(U_c, Y_fc))
        Pk3 = Pk3.reshape(-1, 1)

        Pk4 = - gamma * np.dot((U.T),(np.dot(P_zs, Y_hat[:,k]) - np.dot(P_zs, Y_hat[:,k])))
        Pk4 = Pk4.reshape(-1, 1)

        Pk  = np.vstack((Pk1, Pk2, Pk3, Pk4))
        print('this is PK _________________________________')
        print(Pk)
        
            
        # d_x = np.array([
        #     [-1],
        #     [0],
        #     [1],
        #     [0]
        # ])
        # d_y = np.array([
        #     [0],
        #     [-1],
        #     [0],
        #     [1]
        # ])
        # for i in range(N):
        #     D_k[4*i:4*(i+1), i] = d_x.transpose()
        #     D_k[4*i:4*(i+1), N+i] = d_y.transpose()






nextmsteps(T= 1, N=24, x=10, x1=10, x2=10, y=10, y1=10, y2=10, h=10, g=10, X_fc =0, Y_fc=0, alpha = 1, beta = 1, gamma = 1, m = 2)