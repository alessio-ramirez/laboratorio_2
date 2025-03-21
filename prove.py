from guezzi import *
x = create_dataset([1, 2, 3, 4, 5], 0.1)
y = create_dataset([23, 34, 44, 51, 64], 1)
def linear_func(beta, x):
    a, b = beta
    return a + b * x
stats = perform_fit(x, y, linear_func, p0=4)