# from functions import u_ab, u_ad, u_bc, u_cd
import numpy as np
from deepcopyall import deepcopy
from numpy import linalg as LA


def mkr(n, eps, omega, U_start):
    err = 1
    it = 0
    ERR = []
    U = deepcopy(U_start)
    U_last = deepcopy(U_start)
    U_relax = deepcopy(U_start)
    print(U)
    print(U_last)


    while err > eps:
        # print(U1)
        # print(U)
        for i in range(1, n):
            for j in range(1, n):
                print(U_last[i][j])
                U_relax[i][j] = 0.25*(U[i + 1][j] + U[i][j - 1] + U[i][j + 1] + U[i - 1][j])
                U[i][j] = U_last[i][j] + omega * (U_relax[i][j] - U_last[i][j])
                print('U_last ', U_last[i][j])
                print('U ', U[i][j])

        #
        err = LA.norm(U-U_last, np.inf)
        print(err)
        U_last = deepcopy(U)
        it += 1

    # print(err)
    return U, it




