import numpy as np

def qr_iteration(A, max_iterations=1000, tolerance=1e-6):
    n = A.shape[0]
    eigenvalues = np.zeros(n, dtype=np.complex128)
    eigenvectors = np.eye(n, dtype=np.complex128)

    for _ in range(max_iterations):
        Q, R = np.linalg.qr(A)
        A_new = R @ Q
        eigenvalues_new = np.diag(A_new)
        eigenvectors_new = eigenvectors @ Q

        if np.allclose(eigenvalues, eigenvalues_new, atol=tolerance):
            break

        A = A_new
        eigenvalues = eigenvalues_new
        eigenvectors = eigenvectors_new

    return eigenvalues, eigenvectors

A = np.array([[3, 4, -1, 0],
              [2, -1, 4, 5],
              [0, -1, 7, 6],
              [2, 8, 11, 14]])

eigenvalues, eigenvectors = qr_iteration(A)

# Print the eigenvalues and eigenvectors
print("Eigenvalues:")
print(eigenvalues)

print("Eigenvectors:")
print(eigenvectors)
