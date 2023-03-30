import math
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd



dx, N = 0.1, 10
dt = 10 ** (-2)
K = 50

massive_x = np.arange(0, 1+dx, dx)
massive_t = np.arange(0, 0.5+dt, dt)




def F(X):
    return 1 + X ** 2


def f(X):
    return (X + 0.2) * math.sin(math.pi*X*0.5)


def U_0():
    U_start = [[0.0 for _ in range(N+1)] for _ in range(K+1)] # N столбцов, K строк
    for k in range(K+1):
        U_start[k][0] = 0
        U_start[k][N] = 1.2 * (massive_t[k] + 1)
    for j in range(1, N):
        U_start[0][j] = f(massive_x[j])  # (massive_x[j] + 0.2) * math.sin(math.pi * massive_x[j] * 0.5)
    for j in range(1, N):
        U_start[1][j] = f(massive_x[j]) + dt * F(massive_x[j]) + 0.5 * (dt ** 2) * (U_start[0][j - 1] - 2 * U_start[0][j] + U_start[0][j + 1]) / (dx ** 2)
    return U_start


def Method1(U):
    for i in range(1, K):
        for j in range(1, N):
            U[i+1][j] = (dt ** 2) * (U[i][j+1] - 2 * U[i][j] + U[i][j-1]) / (dx ** 2) + 2 * U[i][j] - U[i-1][j]
    return U


def Thomas(Koeff, D):
    alpha = [0 for _ in range(N+1)]
    betta = [0 for _ in range(N+1)]
    X = [0 for _ in range(N+1)]
    Y = [0 for _ in range(N+1)]

    Y[0] = Koeff[0][0]
    alpha[0] = - Koeff[0][1] / Y[0]
    betta[0] = - D[0] / Y[0]

    for i in range(1, N):
        Y[i] = Koeff[i][i] + Koeff[i][i - 1] * alpha[i - 1]
        alpha[i] = - Koeff[i][i+1] / Y[i]
        betta[i] = (D[i] - Koeff[i][i - 1] * betta[i - 1]) / Y[i]

    Y[N] = Koeff[N][N] + Koeff[N][N - 1] * alpha[N - 1]
    betta[N] = (D[N] - Koeff[N][N - 1] * betta[N - 1]) / Y[N]

    X[N] = betta[N]

    for i in range(N-1, -1, -1):
        X[i] = alpha[i] * X[i + 1] + betta[i]

    return X


def Method2(U):
    # коэффициенты для неявного метода:
    A = 1 / (dx ** 2)
    B = (dx ** 2 + 2 * (dt ** 2)) / ((dt ** 2) * (dx ** 2))
    C = A

    Matrix_koeff = [[0 for _ in range(N+1)] for _ in range(N+1)]
    Matrix_koeff[0][0] = 1
    Matrix_koeff[N][N] = 1
    for i in range(1, N):
        Matrix_koeff[i][i - 1] = -A
        Matrix_koeff[i][i] = B
        Matrix_koeff[i][i + 1] = -C

    F = [0.0 for _ in range(N + 1)]

    for i in range(2, K + 1):
        F[0] = U[i][0]
        F[N] = U[i][N]
        for j in range(1, N):
            F[j] = (2 * U[i-1][j] - U[i - 2][j]) / (dt ** 2)
        U_result = Thomas(Matrix_koeff, F)
        for k in range(N):
            U[i][k] = U_result[k]
    return U


U_1 = Method1(U_0())
U_2 = Method2(U_0())
# _ = {f'{_}': U_2[_] for _ in range(len(U_2))}

 df = pd.DataFrame(_)
# df.to_csv('excel_name.csv', sep=';')
# pe.save_as(array=U_1, dest_file_name='array_data.xlsx')
# pd.DataFrame(U_1, columns=['x = 0', 'x = 0.1', 'x = 0.2', 'x = 0.3', 'x = 0.4', 'x = 0.5', 'x = 0.6', 'x = 0.7', 'x = 0.8', 'x = 0.9', 'x = 1'])
# U_1.to_excel('array_data.xlsx', sheet_name="Лист 1")
# df = pd.DataFrame(np.random.randn(1000, 4), columns=['A', 'B', 'C', 'D'])
# df.head()

# print(U_1)

def printMatrix(Matrix):
    for m in range(0, len(Matrix)):
        for n in range(0, len(Matrix[m])):
            print(round(Matrix[m][n], 4), end='  ')
        print()
    print()


# printMatrix(U_0())
#print('Явный метод:')
#printMatrix(U_1)
print('Неявный метод:')
printMatrix(U_2)


"""print(massive_t[3])
print(massive_t[10])
print(massive_t[15])
print(massive_t[40])
"""
fig = make_subplots(rows=2, cols=2, subplot_titles=('t=0.03', 't=0.1', 't=0.15', 't=0.4'))
fig.add_trace(go.Scatter(x=massive_x, y=U_1[3], name="Явный&nbsp;0.002"), 1, 1)
fig.add_trace(go.Scatter(x=massive_x, y=U_2[3], name="Неявный&nbsp;0.002"), 1, 1)
fig.add_trace(go.Scatter(x=massive_x, y=U_1[10], name="Явный&nbsp;0.004"), 1, 2)
fig.add_trace(go.Scatter(x=massive_x, y=U_2[10], name="Неявный&nbsp;0.004"), 1, 2)
fig.add_trace(go.Scatter(x=massive_x, y=U_1[15], name="Явный&nbsp;0.006"), 2, 1)
fig.add_trace(go.Scatter(x=massive_x, y=U_2[15], name="Неявный&nbsp;0.006"), 2, 1)
fig.add_trace(go.Scatter(x=massive_x, y=U_1[40], name="Явный&nbsp;0.01"), 2, 2)
fig.add_trace(go.Scatter(x=massive_x, y=U_2[40], name="Неявный&nbsp;0.01"), 2, 2)
fig.update_layout(title='Зависимость смещения от координаты в момент времени')

fig.update_xaxes(title='X', row=1, col=1)
fig.update_xaxes(title='X', row=1, col=2)
fig.update_xaxes(title='X', row=2, col=1)
fig.update_xaxes(title='X', row=2, col=2)

fig.update_yaxes(title='U', row=1, col=1)
fig.update_yaxes(title='U', row=1, col=2)
fig.update_yaxes(title='U', row=2, col=1)
fig.update_yaxes(title='U', row=2, col=2)

fig.show()


fig5 = go.Figure(data=[go.Surface(z=U_1, x=massive_x, y=massive_t, name="Явный&nbsp;метод")])
fig5.update_layout(title="Явный&nbsp;метод")
fig5.show()


fig6 = go.Figure(data=[go.Surface(z=U_2, x=massive_x, y=massive_t, name="Неявный&nbsp;метод")])
fig6.update_layout(title="Неявный&nbsp;метод")
fig6.show()