import numpy as np


'''

'''
nodes = [
    np.array([-8., 6.]),
    np.array([-8., 0.]),
    np.array([-11., 0.]),
    np.array([-11., 6.]),
    np.array([-14., 6.]),
    np.array([-14., 0.]),
    np.array([-17., 0.]),
    np.array([-17., 6.]),
    np.array([-5., 0.]),
    np.array([-2., 0.]),
    np.array([1., 6.]),
    np.array([4., 0.]),
    np.array([1., 0.]),
    np.array([2., 6.]),
    np.array([-5., 6.]),
    np.array([-20., 0.]),
]

elements = [
    (1, 2),
    (3, 2),
    (4, 3),
    (5, 4),
    (6, 5),
    (7, 6),
    (8, 7),
    (6, 8),
    (4, 1),
    (2, 4),
    (2, 9),
    (9, 10),
    (11, 10),
    (12, 11),
    (13, 12),
    (11, 13),
    (14, 11),
    (10, 14),
    (15, 2),
    (15, 14),
    (6, 3),
    (4, 6),
    (8, 5),
    (8, 15),
    (15, 7),
    (1, 15),
    (15, 9),
    (10, 15),
    (10, 13),
]

bcs = [11, 15]
forces = [(2, -1e3)]
E = 200000000000
A = 0.0001


def calc_stiffness(element):
    node1 = nodes[element[0]]
    node2 = nodes[element[1]]
    l = np.linalg.norm(node1 - node2)
    l12 = (node2[0] - node1[0]) / l
    m12 = (node2[1] - node1[1]) / l
    tr = np.array([[l12, m12, 0, 0], [0, 0, l12, m12]])
    k_local = E * A / l * np.array([[1, -1], [-1, 1]])
    return np.matmul(np.matmul(tr.T, k_local.T), tr)


def calc_global_stiffnes():
    k = np.zeros((2 * len(nodes), 2 * len(nodes)))
    for el in elements:
        stiffness = calc_stiffness(el)
        for i in range(2):
            for j in range(2):
                k[2 * el[i], 2 * el[j]] += stiffness[2 * i, 2 * j]
                k[2 * el[i], 2 * el[j] + 1] += stiffness[2 * i, 2 * j + 1]
                k[2 * el[i] + 1, 2 * el[j]] += stiffness[2 * i + 1, 2 * j]
                k[2 * el[i] + 1, 2 * el[j] + 1] += stiffness[2 * i + 1, 2 * j + 1]
    return k


def calc_forces():
    f = np.zeros(2 * len(nodes))
    for force in forces:
        f[2 * force[0] + 1] = force[1]
    return f


def calc_bcs(stiffness, forces):
    mod_stiffness = np.copy(stiffness)
    mod_forces = np.copy(forces)
    for bc in bcs:
        mod_stiffness[2 * bc, :] = 0
        mod_stiffness[2 * bc + 1, :] = 0
        mod_stiffness[2 * bc, 2 * bc] = 1
        mod_stiffness[2 * bc + 1, 2 * bc + 1] = 1
        mod_forces[2 * bc] = 0
        mod_forces[2 * bc + 1] = 0
    return mod_stiffness, mod_forces


stiffness = calc_global_stiffnes()
forces = calc_forces()

eq_stiffness, eq_forces = calc_bcs(stiffness, forces)
print(eq_stiffness)
print(np.linalg.det(eq_stiffness))
u = np.linalg.solve(eq_stiffness, eq_forces)
for i in range(0, len(nodes)):
    print("{:.8E} {:.8E}".format(u[2 * i], u[2 * i + 1]))

print(len(u))
h = 0

for el in elements:
    node1 = nodes[el[0]]
    node2 = nodes[el[1]]
    u1 = np.array([u[2 * el[0]], u[2 * el[0] + 1]])
    u2 = np.array([u[2 * el[1]], u[2 * el[1] + 1]])

    l = np.linalg.norm(node1 - node2)
    force = E * A * (np.dot(node1 + u1 - node2 - u2, node1 - node2) / l / l - 1)
    print("{: .8E}".format(force))
    h += 1

print(h)

'''
def calc_stiffness(element):
    node1 = nodes[element[0]]
    print(element[0])
    print(element[1])
    print(node1)
    node2 = nodes[element[1]]

    l = np.linalg.norm(node1 - node2)
    l12 = (node2[0] - node1[0]) / l
    m12 = (node2[1] - node1[1]) / l
    tr = np.array([[l12, m12, 0, 0], [0, 0, l12, m12]])
    k_local = E * A / l * np.array([[1, -1], [-1, 1]])
    return np.matmul(np.matmul(tr.T, k_local.T), tr)


def calc_global_stiffnes():
    k = np.zeros((2 * len(nodes), 2 * len(nodes)))
    for el in elements:
        stiffness = calc_stiffness(el)
        for i in range(2):
            for j in range(2):
                k[2 * el[i], 2 * el[j]] += stiffness[2 * i, 2 * j]
                k[2 * el[i], 2 * el[j] + 1] += stiffness[2 * i, 2 * j + 1]
                k[2 * el[i] + 1, 2 * el[j]] += stiffness[2 * i + 1, 2 * j]
                k[2 * el[i] + 1, 2 * el[j] + 1] += stiffness[2 * i + 1, 2 * j + 1]
    return k


def calc_forces():
    f = np.zeros(2 * len(nodes))
    for force in forces:
        f[2 * force[0] + 1] = force[1]
    return f


def calc_bcs(stiffness, forces):
    mod_stiffness = np.copy(stiffness)
    mod_forces = np.copy(forces)
    for bc in bcs:
        mod_stiffness[2 * bc, :] = 0
        mod_stiffness[2 * bc + 1, :] = 0
        mod_stiffness[2 * bc, 2 * bc] = 1
        mod_stiffness[2 * bc + 1, 2 * bc + 1] = 1
        mod_forces[2 * bc] = 0
        mod_forces[2 * bc + 1] = 0
    return mod_stiffness, mod_forces


stiffness = calc_global_stiffnes()
forces = calc_forces()
eq_stiffness, eq_forces = calc_bcs(stiffness, forces)
u = np.linalg.solve(eq_stiffness, eq_forces)
for i in range(0, len(nodes)):
    print("{:.8E} {:.8E}".format(u[2 * i], u[2 * i + 1]))
for el in elements:
    node1 = nodes[el[0]]
    node2 = nodes[el[1]]
    u1 = np.array([u[2 * el[0]], u[2 * el[0] + 1]])
    u2 = np.array([u[2 * el[1]], u[2 * el[1] + 1]])
    l = np.linalg.norm(node1 - node2)
    force = E * A * (np.dot(node1 + u1 - node2 - u2, node1 - node2) / l / l - 1)
    print("{: .8E}".format(force))

'''