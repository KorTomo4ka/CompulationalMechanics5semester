import numpy as np
from numpy import linalg as LA
import plotly.graph_objs as go


n = 4
X = np.zeros((n, n))
A = np.zeros((n, n))
a = -1
b = 1
dx = 0.01
N = int((b - a) / dx)
print(N)

x = np.linspace(a, b, num=N)
print(dx)

for j in range(n):
        X[0][j] = a**j
        X[1][j] = j * a ** abs(j - 1)
        X[2][j] = b ** j
        X[3][j] = j*b**(j-1)

print(X)

A = LA.inv(X)

print(A)

i = lambda x: A[0][0] + A[1][0] * x + A[2][0] * x**2 + A[3][0] * x**3
j = lambda x: A[0][2] + A[1][2] * x + A[2][2] * x**2 + A[3][2] * x**3
i_tetta = lambda x: A[0][1] + A[1][1] * x + A[2][1] * x**2 + A[3][1] * x**3
j_tetta = lambda x: A[0][3] + A[1][3] * x + A[2][3] * x**2 + A[3][3] * x**3

N_i = i(x)
N_j = j(x)
Ni_tetta = i_tetta(x)
Nj_tetta = j_tetta(x)
Check = np.around(N_j + N_i)
print(Check)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=N_i, name='U1'))
fig.add_trace(go.Scatter(x=x, y=N_j, name='U2'))
fig.update_layout(title='Смещение')
fig.show()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=x, y=Ni_tetta, name='tetta1'))
fig1.add_trace(go.Scatter(x=x, y=Nj_tetta, name='tetta2'))
fig1.update_layout(title='Угол')
fig1.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x, y=Check))
fig2.update_layout(title='График суммы форм')
fig2.show()
