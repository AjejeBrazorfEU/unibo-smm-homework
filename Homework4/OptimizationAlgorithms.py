import numpy as np
from tqdm import tqdm

DIVERGE_THRESHOLD = 1e15

"""
Input:
    l: the function l(w; D) we want to optimize. It is supposed to be a Python function, not an array.
    grad_l: the gradient of l(w; D). It is supposed to be a Python function, not an array.
    w0: an n-dimensional array which represents the initial iterate. By default, it should be randomly sampled.
    data: a tuple (x, y) that contains the two arrays x and y, where x is the input data, y is the output data.
    batch_size: an integer. The dimension of each batch. Should be a divisor of the number of data.
    n_epochs: an integer. The number of epochs you want to reapeat the iterations.
Output:
    w: an array that contains the value of w_k FOR EACH iterate w_k (not only the latter).
    f_val: an array that contains the value of l(w_k; D)
        FOR EACH iterate w_k ONLY after each epoch.
    grads: an array that contains the value of grad_l(w_k; D)
        FOR EACH iterate w_k ONLY after each epoch.
    err: an array the contains the value of ||grad_l(w_k; D)||_2
        FOR EACH iterate w_k ONLY after each epoch
"""


def SGD(
    l,
    grad_l,
    w0,
    data,
    batch_size: int = 100,
    n_epochs: int = 3,
    alpha=10e-8,
    VERBOSITY=1,
    lam=1,
):
    X, Y = data
    N = len(Y)
    # Creating the return values
    w, f_val, grads, err = [], [], [], []
    w.append(w0)
    k = 0
    for epoch in (
        tqdm(range(n_epochs), desc=f"SGD") if VERBOSITY == 1 else range(n_epochs)
    ):
        # for epoch in range(n_epochs):
        indices = np.arange(len(Y))

        # Shuffle the indices
        np.random.shuffle(indices)
        while len(indices) > 0:
            indices_batch = indices[:batch_size]
            X_batch = X[indices_batch]
            Y_batch = Y[indices_batch]
            # Update the w value using the gradient calculated only on the batch
            w.append(w[k] - alpha * grad_l(w[k], X_batch.T, Y_batch, lam=lam))
            k += 1
            if np.linalg.norm(w[k], 2) > DIVERGE_THRESHOLD:
                return w, f_val, grads, err, k, False
            # Remove the used indices
            indices = indices[batch_size:]
        # Adding the return values
        f_val.append(l(w[k], X.T, Y, lam=lam))
        grads.append(grad_l(w[k], X.T, Y, lam=lam))
        err.append(np.linalg.norm(grads[epoch], 2))

    return w, f_val, grads, err, k, True


DIVERGE_THRESHOLD = 1e15
"""
Returns the following values:
    x: ndarray. Array of iterate.
    k: int. Number of iterations.
    f_val: ndarray. Array of f(x) values.
    grads: ndarray. Array of gradient values.
    err: ndarray. Array of error values.
    converge: bool. True if the method converges, False otherwise.
"""


def GD(
    f,
    grad_f,
    x0,
    X,
    Y,
    kmax: int = 1000,
    tolf: float = 1e-6,
    tolx: float = 1e-6,
    alpha: float = 0.1,
    VERBOSITY=1,
    lam=1,
):
    # x, f_val, grads, err = np.zeros((kmax, len(x0))), np.zeros((kmax, )), np.zeros((kmax, 785)), np.zeros((kmax, ))
    x, f_val, grads, err = [], [], [], []
    alpha = backtracking(f, grad_f, x0, X, Y, lam=lam)

    # Setting the initial values because the for cycle skips the index 0
    x.append(x0)
    f_val.append(f(x0, X.T, Y, lam=lam))
    grads.append(grad_f(x0, X.T, Y, lam=lam))
    err.append(np.linalg.norm(grads[0], 2))

    for k in tqdm(range(1, kmax), desc=f"GD") if VERBOSITY == 1 else range(1, kmax):
        # Update the x value iterativly and saves the last value
        x.append(x[k - 1] - (alpha * grad_f(x[k - 1], X.T, Y, lam=lam)).T)
        if sum(x[k] > DIVERGE_THRESHOLD) > 0:
            if VERBOSITY == 1:
                print("Diverging")
            return x[:k], k, f_val[:k], grads[:k], err[:k], False

        alpha = backtracking(f, grad_f, x[k], X, Y, lam=lam)
        # print(f'Alpha = {alpha}')

        # Adding the values to be returned
        f_val.append(f(x[k], X.T, Y, lam=lam))
        grads.append(grad_f(x[k], X.T, Y, lam=lam).flatten())
        err.append(np.linalg.norm(grads[k], 2))
        if sum(grads[k] > DIVERGE_THRESHOLD) > 0:
            if VERBOSITY == 1:
                print("Diverging")
            return x[: k + 1], k, f_val[: k + 1], grads[: k + 1], err[: k + 1], False

        # Check the stop condition
        if np.linalg.norm(grad_f(x[k], X.T, Y, lam=lam), 2) < tolf * np.linalg.norm(
            grad_f(x0, X.T, Y, lam=lam), 2
        ):
            if VERBOSITY == 1:
                print("Stopping for function tolerance")
            return x[: k + 1], k, f_val[: k + 1], grads[: k + 1], err[: k + 1], True
        if np.linalg.norm(x[k] - x[k - 1], 2) < tolx:
            if VERBOSITY == 1:
                print("Stopping for x tolerance")
            return x[: k + 1], k, f_val[: k + 1], grads[: k + 1], err[: k + 1], True

    if VERBOSITY == 1:
        print("Reached max iterations")
    return x, kmax - 1, f_val[: k + 1], grads[: k + 1], err[: k + 1], True


def backtracking(f, grad_f, x, X, Y, lam=1):
    """
    This function is a simple implementation of the backtracking algorithm for
    the GD (Gradient Descent) method.

    f: function. The function that we want to optimize.
    grad_f: function. The gradient of f(x).
    x: ndarray. The actual iterate x_k.
    """
    alpha = 1
    c = 0.8
    tau = 0.25

    n = 0
    if grad_f(x, X.T, Y, lam=lam).shape != (1, 1):
        n = np.linalg.norm(grad_f(x, X.T, Y, lam=lam), 2)
    else:
        n = np.abs(grad_f(x, X.T, Y, lam=lam))

    while f(
        (x - (alpha * grad_f(x, X.T, Y, lam=lam)).T).flatten(), X.T, Y, lam=lam
    ) > f(x, X.T, Y, lam=lam) - c * alpha * (n**2):
        alpha = tau * alpha

        if alpha < 1e-10:
            break
    return alpha
