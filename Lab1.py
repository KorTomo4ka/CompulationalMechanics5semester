import math
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


h, N = 0.1, 6
dt = 10 ** (-3)
K = 10

massive_x = np.arange(0, 0.7, h)
massive_t = np.arange(0, 0.011, dt)


def temp0():
    T_start = [[0.0 for _ in range(N+1) ] for _ in range(K+1)] # N столбцов, K строк
    for k in range(K+1):
        T_start[k][0] = 1.4
        T_start[k][6] = massive_t[k] + 1
    for j in range(1, N):
        T_start[0][j] = 1 - math.log((massive_x[j] + 0.4), 10)
    return T_start


def Method1(T):
    for i in range(K):
        for j in range(1, N):
            T[i+1][j] = dt * (T[i][j+1] - 2 * T[i][j] + T[i][j-1]) / (h ** 2) + T[i][j]
    return T


# коэффициенты для неявного метода:
A = 1 / (h ** 2)
B = (h ** 2 + 2 * dt) / (dt * (h ** 2))
C = 1 / (h ** 2)


def Method2(T):
    # прямой ход прогонки
    P = [0.0 for _ in range(N+1)]
    Q = [0.0 for _ in range(N+1)]
    F = [0.0 for _ in range(N+1)]
    P[0] = C / B
    for i in range(N+1):
        F[i] = T[0][i] / dt
    for k in range(K):
        Q[0] = (F[1] + A * T[k+1][0]) / B
        for i in range(1, N+1):
            P[i] = C / (B - A * P[i-1])
            Q[i] = (F[i] + A * Q[i-1])/(B - A * P[i-1])
    # обратный ход прогонки
        for i in range(N-1, 0, -1):
            T[k+1][i] = P[i] * T[k+1][i+1] + Q[i]
        for i in range(N+1):
            F[i] = T[k+1][i] / dt
    return T


T_1 = Method1(temp0())
T_2 = Method2(temp0())


def printMatrix(Matrix):
    for m in range(0, len(Matrix)):
        for n in range(0, len(Matrix[m])):
            print(round(Matrix[m][n], 4), end=' ')
        print()
    print()

# printMatrix(temp0())
# printMatrix(T_1)
# printMatrix(T_2)


fig = make_subplots(rows=2, cols=2, subplot_titles=('t=0.002', 't=0.004', 't=0.006', 't=0.01'))
fig.add_trace(go.Scatter(x=massive_x, y=T_1[3], name="Явный&nbsp;0.002"), 1, 1)
fig.add_trace(go.Scatter(x=massive_x, y=T_2[3], name="Неявный&nbsp;0.002"), 1, 1)
fig.add_trace(go.Scatter(x=massive_x, y=T_1[5], name="Явный&nbsp;0.004"), 1, 2)
fig.add_trace(go.Scatter(x=massive_x, y=T_2[5], name="Неявный&nbsp;0.004"), 1, 2)
fig.add_trace(go.Scatter(x=massive_x, y=T_1[7], name="Явный&nbsp;0.006"), 2, 1)
fig.add_trace(go.Scatter(x=massive_x, y=T_2[7], name="Неявный&nbsp;0.006"), 2, 1)
fig.add_trace(go.Scatter(x=massive_x, y=T_1[10], name="Явный&nbsp;0.01"), 2, 2)
fig.add_trace(go.Scatter(x=massive_x, y=T_2[10], name="Неявный&nbsp;0.01"), 2, 2)
fig.update_layout(title='Зависимость температуры от координаты в момент времени')

fig.update_xaxes(title='X', row=1, col=1)
fig.update_xaxes(title='X', row=1, col=2)
fig.update_xaxes(title='X', row=2, col=1)
fig.update_xaxes(title='X', row=2, col=2)

fig.update_yaxes(title='T', row=1, col=1)
fig.update_yaxes(title='T', row=1, col=2)
fig.update_yaxes(title='T', row=2, col=1)
fig.update_yaxes(title='T', row=2, col=2)

fig.show()

fig5 = go.Figure(data=[go.Surface(z=T_1, x=massive_x, y=massive_t, name="Явный&nbsp;метод")])
fig5.update_layout(title="Явный&nbsp;метод")
fig5.show()

fig6 = go.Figure(data=[go.Surface(z=T_2, x=massive_x, y=massive_t, name="Неявный&nbsp;метод")])
fig6.update_layout(title="Неявный&nbsp;метод")
fig6.show()


