import numpy as np

def is_eigenpair(A, v, lam, tol=1e-9):
    LHS = A @ v
    RHS = lam * v
    error = np.max(np.abs(LHS - RHS))
    return error < tol

def rayleigh_quotient(A, v):
    Av = A @ v
    numerator = v @ Av
    denominator = v @ v
    return numerator / denominator

def power_iteration(A, tol=1e-9, max_iter=1000):
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.sqrt(v @ v)
    lamda_old = 0
    lamda = 0
    for i in range(max_iter):
        w = A @ v
        v = w / np.sqrt(w @ w)
        lamda = rayleigh_quotient(A, v)
        if abs(lamda - lamda_old) < tol:
            break
        lamda_old = lamda
    return lamda, v

def centre(X):
    X = np.array(X, dtype=float)
    X_centred = X - X.mean(axis=0)
    return X_centred

def covariance(X):
    Xc = centre(X)
    m = X.shape[0]
    numerator = Xc.T @ Xc
    denominator = m - 1
    C = numerator / denominator
    return C

def eigen_decompose(A, k):
    A = A.copy()
    eigenvalues = []
    eigenvectors = []
    for i in range(k):
        lam, v = power_iteration(A)
        eigenvalues.append(lam)
        eigenvectors.append(v)
        A = A - lam * np.outer(v , v)
    return eigenvalues, eigenvectors

def pca(X, k):
    C = covariance(X)
    eigenvalues, eigenvectors = eigen_decompose(C, k)

    V = np.array(eigenvectors).T
    Xc = centre(X)
    projected = Xc @ V

    eigenvalues = np.array(eigenvalues)
    ratios = eigenvalues / np.trace(C)
    return eigenvalues, V, ratios, projected

if __name__ == '__main__':

    # ---- Test 1: symmetric, dominant lambda = 3 ----
    A1 = np.array([[2.0, 1.0],
                   [1.0, 2.0]])
    lam1, v1 = power_iteration(A1)
    print("Test 1  lambda =", lam1)
    print("Test 1  v      =", v1)
    print("Test 1  is_eigenpair:", is_eigenpair(A1, v1, lam1, tol=1e-5))

    # ---- Test 2: the trap, dominant lambda = -3 (vector flips each step) ----
    A2 = np.array([[-3.0, 0.0],
                   [ 0.0, 1.0]])
    lam2, v2 = power_iteration(A2)
    print("Test 2  lambda =", lam2)
    print("Test 2  v      =", v2)
    print("Test 2  is_eigenpair:", is_eigenpair(A2, v2, lam2, tol=1e-5))

    # ---- Test 3: oracle on a random symmetric matrix ----
    M = np.random.randn(5, 5)
    M = M + M.T
    lam3, v3 = power_iteration(M)
    eigenvalues = np.linalg.eig(M)[0]
    dominant = eigenvalues[np.argmax(np.abs(eigenvalues))]
    print("Test 3  my lambda   =", lam3)
    print("Test 3  numpy domin =", dominant)
    print("Test 3  match:", abs(lam3 - dominant) < 1e-6)
    print("Test 3  is_eigenpair:", is_eigenpair(M, v3, lam3, tol=1e-5))

    X = np.array([[1.0, 2.0],
                  [3.0, 6.0],
                  [5.0, 10.0]])

    # ---- centre: each column mean subtracted ----
    print("centred:")
    print(centre(X))  # expect [[-2,-4],[0,0],[2,4]]

    # ---- covariance: hand-check ----
    C = covariance(X)
    print("covariance:")
    print(C)  # expect [[4, 8],[8, 16]]

    # ---- oracle ----
    print("numpy cov:")
    print(np.cov(X, rowvar=False))  # should match C
    print("match:", np.allclose(C, np.cov(X, rowvar=False)))

    A = np.array([[2.0, 0.0, 0.0],
                  [0.0, 5.0, 0.0],
                  [0.0, 0.0, 3.0]])

    eigenvalues, eigenvectors = eigen_decompose(A, 3)

    print("my eigenvalues:", eigenvalues)  # expect 5, 3, 2

    oracle = np.linalg.eigh(A)[0][::-1]  # eigh ascending, reversed
    print("numpy eigh:    ", oracle)  # expect [5, 3, 2]

    print("match:", np.allclose(eigenvalues, oracle))

    X = np.array([[1.0, 2.0],
                  [3.0, 6.0],
                  [5.0, 10.0],
                  [2.0, 4.0],
                  [4.0, 8.0]])

    eigenvalues, V, ratios, projected = pca(X, 2)

    print("explained variance:", np.round(eigenvalues, 4))
    print("ratio:             ", np.round(ratios, 4))  # expect [1.0, 0.0]
    print("projected:")
    print(np.round(projected, 4))