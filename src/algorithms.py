import numpy as np


def gd(x0, A, b, alpha, tol):
    xs  = [x0]
    x = x0
    while True:
        g = A @ x - b
        x = x - alpha * g
        if np.linalg.norm(x - xs[-1]) < tol:
            break
        xs.append(x)
    return np.array(xs).squeeze()

def sd(x0, A, b, tol):
    xs  = [x0]
    x = x0
    while True:
        r     = b - A @ x
        alpha = (r.T @ r) / (r.T @ A @ r)
        x     = x + alpha * r
        if np.linalg.norm(x - xs[-1]) < tol:
            break
        xs.append(x)
    return np.array(xs)

def get_conjugate_basis(A):
    vs  = np.random.randn(A.shape[0], A.shape[0])
    v   = vs[0]
    u   = v / np.linalg.norm(v)
    us  = [u]
    for i in range(1, A.shape[0]):
        v = vs[i]
        for j in range(i):
            v -= proj(A, v, us[j-1])
        u = v / np.linalg.norm(v)
        us.append(u)
    return np.array(us)

def proj(A, v, u):
    Au   = A @ u
    beta = (v @ Au) / (u @ Au)
    return beta * u

def cgd_direct(A, b):
    # Run Gram–Schmidt like process to get A-orthogonal basis.
    D = get_conjugate_basis(A)

    # Compute the step size alphas.
    alphas = np.empty(A.shape[0])
    for i in range(A.shape[0]):
        d = D[i]
        alphas[i] = (d @ b) / (d @ A @ d)

    # Directly compute minimum point x_star.
    x_star = np.sum(alphas[:, None] * D, axis=0)
    return np.array([x_star])

def cgd_iter(x0, A, b):
    xs  = [x0]
    ds  = []
    x = x0
    for i in range(A.shape[0]):
        # Compute next search direction.
        r = b - A @ x

        # Remove previous search directions.
        projs = 0
        for k in range(i):
            d_k     = ds[k-1]
            beta_k  = (r.T @ A @ d_k) / (d_k.T @ A @ d_k)
            projs  += beta_k * d_k
        d = r - projs

        # Take step.
        alpha = (r.T @ d) / (d.T @ A @ d)
        x = x + alpha * d

        xs.append(x)
        ds.append(d)
    return np.array(xs)

def cgd(x0, A, b):
    x = x0
    r    = b - A @ x
    d    = r
    rr   = r.T @ r
    xs   = [x]
    for _ in range(A.shape[0]):
        Ad     = A @ d
        alpha  = rr / (d.T @ Ad)
        x      = x + alpha * d
        r      = r - alpha * Ad
        rr_new = r.T @ r
        beta   = rr_new / rr
        d      = r + beta * d
        rr     = rr_new
        xs.append(x)
    return np.array(xs)
