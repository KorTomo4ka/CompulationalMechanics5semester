def grid(a,b,n):
    x = [0 for _ in range (n+1)]
    h = (b - a) / n
    for i in range(n+1):
        x[i] = a + i * h
    return x