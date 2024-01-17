CACHE = {}
import matplotlib.pyplot as plt


def rovina_del_giocatore(k):
    if (k) in CACHE:
        return CACHE[(k)]
    # if k==1:

    if k == 0:
        return 0
    elif k == N:
        return 1
    CACHE[(k)] = (t / c) ** (k - 1) * rovina_del_giocatore(1) + rovina_del_giocatore(
        k - 1
    )
    return CACHE[(k)]


c = 0.55
t = 1 - c
k = 4
N = 50
CACHE[(1)] = 1 / sum([(t / c) ** (i - 1) for i in range(1, N + 1)])
print(CACHE[(1)])
prob = rovina_del_giocatore(k)
print(f"Probabilità che il giocatore vinca partendo con {k} monete = {prob}")

plt.plot(range(1, N + 1), [rovina_del_giocatore(i) for i in range(1, N + 1)])
plt.show()
