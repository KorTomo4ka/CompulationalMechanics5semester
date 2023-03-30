import math

def u_ab(y):
    return 30 * math.sin(math.pi * y)


def u_bc(x):
    return 20 * x


def u_cd(y):
    return 20 * y


def u_ad(x):
    return 30 * x * (1 - x)