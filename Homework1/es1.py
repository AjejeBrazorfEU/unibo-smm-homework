# Given a matrix A ∈ Rn×n and the vector xtrue = (1, 1, . . . , 1)T ∈ Rn, write a script that:
import numpy as np
import matplotlib.pyplot as plt
import scipy

n = 3

A = np.random.rand(n, n)
xtrue = np.ones((n, 1))

# Computes the right-hand side of the linear system b = Axtrue
right_hand_side = np.dot(A, xtrue)

# Computes the condition number in 2-norm of the matrix A. It is ill-conditioned? What if we use
# the ∞-norm instead of the 2-norm?
k_2_A = np.linalg.norm(A, 2) * np.linalg.norm(np.linalg.inv(A), 2)
k_inf_A = np.linalg.norm(A, np.inf) * np.linalg.norm(np.linalg.inv(A), np.inf)

print("k_2_A = ", k_2_A)
print("k_inf_A = ", k_inf_A)

# Solves the linear system Ax = b with the function np.linalg.solve()
x = np.linalg.solve(A, right_hand_side)
print("x = ", x)

# Computes the relative error between the solution computed before and the true solution xtrue
E = np.linalg.norm(x - xtrue, 2) / np.linalg.norm(xtrue, 2)
print("E = ", E)


# Plot a graph (using matplotlib.pyplot) with the relative errors as a function of n and (in a new
# window) the condition number in 2-norm K2(A) and in ∞-norm, as a function of n.

# A random matrix (created with the function np.random.rand()) with size varying with n =
# {10, 20, 30, . . . , 100}
n = np.arange(10, 110, 10)
E = np.zeros(len(n))
for i in n:
    A = np.random.rand(i, i)
    xtrue = np.ones((i, 1))
    right_hand_side = np.dot(A, xtrue)
    x = np.linalg.solve(A, right_hand_side)
    E[i // 10 - 1] = np.linalg.norm(x - xtrue, 2) / np.linalg.norm(xtrue, 2)

plt.figure(1)
plt.plot(n, E)
plt.title("Relative error with random matrix")
plt.xlabel("n")
plt.ylabel("E")

# The Vandermonde matrix (np.vander) of dimension n = {5, 10, 15, 20, 25, 30} with respect to the
# vector x = {1, 2, 3, . . . , n}

n = np.arange(5, 35, 5)
E = np.zeros(len(n))
for i in n:
    xx = np.arange(1, i + 1, 1)
    A = np.vander(xx, i)
    xtrue = np.ones((i, 1))
    right_hand_side = np.dot(A, xtrue)
    x = np.linalg.solve(A, right_hand_side)
    E[i // 5 - 1] = np.linalg.norm(x - xtrue, 2) / np.linalg.norm(xtrue, 2)
print("Vander errors: ", E)
plt.figure(2)
plt.plot(n, E)
plt.title("Relative error with vander matrix")
plt.xlabel("n")
plt.ylabel("E")


# The Hilbert matrix (scipy.linalg.hilbert) of dimension n = {4, 5, 6, . . . , 12}
n = np.arange(4, 13, 1)
E = np.zeros(len(n))
for i in n:
    A = scipy.linalg.hilbert(i)
    xtrue = np.ones((i, 1))
    right_hand_side = np.dot(A, xtrue)
    x = np.linalg.solve(A, right_hand_side)
    E[i - 4] = np.linalg.norm(x - xtrue, 2) / np.linalg.norm(xtrue, 2)

print(E)
plt.figure(3)
plt.plot(n, E)
plt.title("Relative error with hilbert matrix")
plt.xlabel("n")
plt.ylabel("E")
plt.show()
