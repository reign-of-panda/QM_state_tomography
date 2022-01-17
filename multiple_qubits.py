import numpy as np
from copy import deepcopy

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

def get_indx(N):
	to_return = []
	pauli_indx = [0 for i in range(N)]
	for n in range(4**N):
		to_return.append(deepcopy(pauli_indx))
		to_add = -1
		while True:
			if to_add == -N - 1:
				break
			elif pauli_indx[to_add] == 3:
				pauli_indx[to_add] = 0
				to_add -= 1
			else:
				pauli_indx[to_add] += 1
				break
	return to_return

# Select number of qubits
N = 2

# Pauli matrices
pauli_0 = np.identity(2)
pauli_1 = np.array([[0, 1], [1, 0]])
pauli_2 = np.array([[0, -1j], [1j, 0]])
pauli_3 = np.array([[1, 0], [0, -1]])
pauli = np.array([pauli_0, pauli_1, pauli_2, pauli_3])

sigma = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
pauli_indx = get_indx(N)
for n in range(4**N):
	to_append = np.array([1])
	for i in range(N):
		to_append = np.kron(to_append, pauli[pauli_indx[n][i]])
	sigma[n] = to_append

# Dynamics
phi = 0
U = np.cos(phi) * np.identity(2**N) + 1j * np.sin(phi) * sigma[2]

# Initial states
s0 = np.array([[1], [0]])
s1 = np.array([[0], [1]])
s2 = np.array([[1], [1]]) / np.sqrt(2)
s3 = np.array([[1], [1j]]) / np.sqrt(2)
s = np.array([s0, s1, s2, s3])

rho = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
for n in range(4**N):
	to_append = np.array([1])
	for i in range(N):
		to_append = np.kron(to_append, s[pauli_indx[n][i]])
	rho[n] = to_append @ to_append.conj().T

# Projectors
P0 = np.array([[1], [0]])
P1 = np.array([[0], [1]])
P2 = np.array([[1], [1]]) / np.sqrt(2)
P3 = np.array([[1], [1j]]) / np.sqrt(2)
P_single = np.array([P0, P1, P2, P3])

P = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
for n in range(4**N):
	to_append = np.array([1])
	for i in range(N):
		to_append = np.kron(to_append, P_single[pauli_indx[n][i]])
	P[n] = to_append @ to_append.conj().T

# Finding A
A = np.zeros((4**N, 4**N, 4**N, 4**N), dtype = np.complex128)
A_unpacked = np.zeros((16**N, 16**N), dtype = np.complex128)
for p in range(4**N):
	for q in range(4**N):
		for i in range(4**N):
			for j in range(4**N):
				temp = np.trace(P[q] @ sigma[i] @ rho[p] @ sigma[j].conj().T)
				A[p, q, i, j] = temp
				A_unpacked[(4**N) * p + q, (4**N) * i + j] = temp

# Finding rho_final
rho_final = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
for i in range(4**N):
	rho_final[i] = U @ rho[i] @ U.conj().T

# Finding P_pq given the projectors and final states
Prob = np.zeros((4**N, 4**N), dtype = np.complex128)
Prob_unpacked = np.zeros((16**N, ), dtype = np.complex128)
for p in range(4**N):
	for q in range(4**N):
		temp = np.trace(P[q] @ rho_final[p])
		Prob[p, q] = temp
		Prob_unpacked[(4**N) * p + q] = temp

# Given P_pq and A_pqij, find eta_ij
eta_unpacked = np.linalg.solve(A_unpacked, Prob_unpacked)
eta = eta_unpacked.reshape((4**N, 4**N))

print(np.round(eta, 5))
print()
print_latex(np.round(eta, 5))