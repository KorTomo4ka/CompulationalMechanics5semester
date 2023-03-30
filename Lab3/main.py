import math
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from Grid import grid
from MKR import mkr
from functions import u_bc, u_cd, u_ab, u_ad

ax = 0
bx = 1
ay = 0
by = 1
n = 5

massive_x = grid(ax, bx, n)
massive_y = grid(ay, by, n)


It = []
Omega = []
eps = 0.0001

U_0 = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]

for k in range(n + 1):
    U_0[n - k][0] = u_ab(massive_y[k])
    U_0[n - k][n] = u_cd(massive_y[k])
for j in range(1, n):
    U_0[0][j] = u_bc(massive_x[j])
    U_0[n][j] = u_ad(massive_x[j])


def printMatrix(Matrix):
    for m in range(0, len(Matrix)):
        for n in range(0, len(Matrix[m])):
            print(round(Matrix[m][n], 2), end='  ')
        print()
    print()

printMatrix(U_0)
"""
for omega in range(1, 3):
    omg = omega / 10
    U, it = mkr(n, eps, omg, U_0)
    It.append(it)
    Omega.append(omg)
    # printMatrix(U)
"""



print(It)
print(Omega)


"""
#print('Явный метод:')
#printMatrix(U_1)
print('Неявный метод:')
printMatrix(U_2)


print(massive_t[3])
print(massive_t[10])
print(massive_t[15])
print(massive_t[40])

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
"""