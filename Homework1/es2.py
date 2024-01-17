"""
The Machine epsilon is the distance between 1 and the next floating point number. Compute , which
is defined as the smallest floating point number such that it holds:
f l(1 + epsilon) > 1
Tips: use a while structure.
"""

epsilon = 1
while 1 + epsilon > 1:
    epsilon = epsilon/2

print("epsilon = ", epsilon * 2)


"""
Let’s consider the sequence 
an = (1 + 1/n )^n.
It is well known that: 
lim n→∞ an = e
where e is the Euler costant. Choose different values for n, compute an and compare it to the real
value of the Euler costant. What happens if you choose a large value of n? Guess the reason."""
import numpy as np
import matplotlib.pyplot as plt
def an(n):
    return (1 + 1/n)**n

n = np.arange(1e15,1e16,1e15)
plt.figure(1)
sequence = an(n)
error = np.abs(sequence - np.exp(1))
plt.plot(n,error)
plt.title("Error of the sequence")
plt.xlabel("n")
plt.ylabel("Error")
#plt.show()

"""
Let’s consider the matrices:
"""
A = np.array([[4,2],[1,3]])
B = np.array([[4,2],[2,1]])

"""
Compute the rank of A and B and their eigenvalues. Are A and B full-rank matrices? Can you infer
some relationship between the values of the eigenvalues and the full-rank condition? Please, corroborate
your deduction with other examples.
"""
rA = np.linalg.matrix_rank(A)
rB = np.linalg.matrix_rank(B)
print("Rank of A = ", rA)
print("Rank of B = ", rB)

eigValA,eigVecA = np.linalg.eig(A)
eigValB,eigVecB = np.linalg.eig(B)
print("Eigenvalues of A = ", eigValA)
print("Eigenvectors of A = ", eigVecA)
print("Eigenvalues of B = ", eigValB)
print("Eigenvectors of B = ", eigVecB)

# A is full rank, B is not full rank because row 1 and row 2 are linearly dependent
detA = np.linalg.det(A)
detB = np.linalg.det(B)
print("Determinant of A = ", detA)
print("Determinant of B = ", detB)
