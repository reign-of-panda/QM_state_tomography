import numpy as np

def print_latex(arr):
	latex = "\\begin{pmatrix}"
	dim = arr.shape

	if len(dim) == 2:
		for i in range(dim[0]):
			for j in range(dim[1]):
				str_ij = str(arr[i, j])
				if str_ij[-1] == ".":
					str_ij = str_ij[:-1]
				latex += str_ij + "&"
				if j == dim[1] - 1:
					latex = latex[: -1] + "\\\\ "
		latex = latex[:-3]
		latex += "\\end{pmatrix}"

	elif len(dim) == 1:
		for i in range(dim[0]):
			str_i = str(arr[i])
			if str_i[-1] == ".":
				str_i = str_i[:-1]
			latex += str_i + "\\\\ "
		latex = latex[:-3]
		latex += "\\end{pmatrix}"

	print(latex)

# Pauli matrices
sigma_0 = np.identity(2)
sigma_1 = np.array([[0, 1], [1, 0]])
sigma_2 = np.array([[0, -1j], [1j, 0]])
sigma_3 = np.array([[1, 0], [0, -1]])
sigma = np.array([sigma_0, sigma_1, sigma_2, sigma_3])

# Dynamics
phi = 1
U = np.cos(phi) * np.identity(2) + 1j * np.sin(phi) * sigma_2

# Initial states
s0 = np.array([[1], [0]])
s1 = np.array([[0], [1]])
s2 = np.array([[1], [1]]) / np.sqrt(2)
s3 = np.array([[1], [1j]]) / np.sqrt(2)

rho_0 = s0 @ s0.conj().T
rho_1 = s1 @ s1.conj().T
rho_2 = s2 @ s2.conj().T
rho_3 = s3 @ s3.conj().T
rho = np.array([rho_0, rho_1, rho_2, rho_3])

# Projectors
P0 = s0 @ s0.conj().T
P1 = s1 @ s1.conj().T
P2 = s2 @ s2.conj().T
P3 = s3 @ s3.conj().T
P = np.array([P0, P1, P2, P3])

# Finding A
A = np.zeros((4, 4, 4, 4), dtype = np.complex128)
A_unpacked = np.zeros((16, 16), dtype = np.complex128)
for p in range(4):
	for q in range(4):
		for i in range(4):
			for j in range(4):
				temp = np.trace(P[q] @ sigma[i] @ rho[p] @ sigma[j].conj().T)
				A[p, q, i, j] = temp
				A_unpacked[4 * p + q, 4 * i + j] = temp

# Finding rho_final
rho_final = np.zeros((4, 2, 2), dtype = np.complex128)
for i in range(4):
	rho_final[i] = U @ rho[i] @ U.conj().T

# Finding P_pq given the projectors and final states
Prob = np.zeros((4, 4), dtype = np.complex128)
Prob_unpacked = np.zeros((16, ), dtype = np.complex128)
for p in range(4):
	for q in range(4):
		temp = np.trace(P[q] @ rho_final[p])
		Prob[p, q] = temp
		Prob_unpacked[4 * p + q] = temp

# Given P_pq and A_pqij, find eta_ij
eta_unpacked = np.linalg.solve(A_unpacked, Prob_unpacked)
eta = eta_unpacked.reshape((4, 4))

print(np.round(eta, 5))
print()
print_latex(np.round(eta, 5))