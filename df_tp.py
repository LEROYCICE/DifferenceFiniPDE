import numpy as np
import matplotlib.pyplot as plt

# On veut résoudre -u'' = f sur ]0, 1[ numeriquement
# conditions au bord de type Dirichlet: u(0) = u(1) = 0
#

def matrice_A(n):

    A = 2 * np.eye(n-1) - np.eye(n-1, k = 1) - np.eye(n-1, k = -1)

    return A

def solution_exacte(x):

    return np.sin(2 * np.pi * x)

def f_poisson(x):

    return (2 * np.pi)**2 * np.sin(2 * np.pi *x)

def resolution_poisson(n):
    h = 1/n
    x = np.linspace(0,1,n+1)

    A = matrice_A(n)
    F = f_poisson(x[1:-1])

    U_interieur = np.linalg.solve(A, h**2 * F)

    U = np.zeros(n+1)

    U[1:-1] = U_interieur

    return x, U

# Graphiques

plt.figure(figsize=(12, 8))

n_valeurs = [10, 30, 50]

x_exact = np.linspace(0, 1, 100)

U_exact = solution_exacte(x_exact)

plt.subplot(1, 2, 1)

plt.plot(x_exact, U_exact, label = "Solution exacte")

for n in n_valeurs:
    x, U = resolution_poisson(n)

    plt.plot(x, U,'o--',label = f'n = {n}')

plt.xlabel('x')
plt.ylabel('U(x)')
plt.legend()
plt.title('Solution Approchée')
plt.grid(True)

# Exercice 3

def calcul_erreur():
    k_valeurs = range(1,12)

    n_valeurs = [2**k for k in k_valeurs]

    h_valeurs = [1/n for n in n_valeurs]

    erreur_l2 = []

    erreur_inf = []

    for n in n_valeurs:
        x, U_appox = resolution_poisson(n)

        U_exact = solution_exacte(x)

        erreur_l2 = np.sqrt(np.sum((U_appox - U_exact)**2)/n)

        erreur_l2 = erreur_l2.append(erreur_l2)

        erreur_inf = np.max(np.abs(U_appox - U_exact))

        erreur_inf = erreur_inf.append(erreur_inf)
    
    return h_valeurs, erreur_inf, erreur_l2

h_vals, err_l2, err_inf = calcul_erreur()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.loglog(h_vals, err_l2, 'bo--', label = "erreur l2")

plt.loglog(h_vals, err_inf, 'ro--', label = "erreur infini")

plt.xlabel('x')

plt.ylabel('Erreur')

plt.grid(True)

plt.show()
