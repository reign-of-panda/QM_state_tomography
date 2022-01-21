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
N = 1

# Pauli matrices
pauli_0 = np.identity(2)
pauli_1 = np.array([[0, 1], [1, 0]])
pauli_2 = np.array([[0, -1j], [1j, 0]])
pauli_3 = np.array([[1, 0], [0, -1]])
pauli = np.array([pauli_0, pauli_1, pauli_2, pauli_3])

def get_sigma(N):
	sigma = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
	pauli_indx = get_indx(N)
	for n in range(4**N):
		to_append = np.array([1])
		for i in range(N):
			to_append = np.kron(to_append, pauli[pauli_indx[n][i]])
		sigma[n] = to_append
	return sigma

# Dynamics
phi = 1
def dynamics_unitary(phi, sigma, N):
	return np.cos(phi) * np.identity(2**N) + 1j * np.sin(phi) * sigma[2]

# Initial states
s0 = np.array([[1], [0]])
s1 = np.array([[0], [1]])
s2 = np.array([[1], [1]]) / np.sqrt(2)
s3 = np.array([[1], [1j]]) / np.sqrt(2)
s = np.array([s0, s1, s2, s3])

def get_rho(N):
	rho = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
	pauli_indx = get_indx(N)
	for n in range(4**N):
		to_append = np.array([1])
		for i in range(N):
			to_append = np.kron(to_append, s[pauli_indx[n][i]])
		rho[n] = to_append @ to_append.conj().T
	return rho

# Projectors
P0 = np.array([[1], [0]])
P1 = np.array([[0], [1]])
P2 = np.array([[1], [1]]) / np.sqrt(2)
P3 = np.array([[1], [1j]]) / np.sqrt(2)
P_single = np.array([P0, P1, P2, P3])

def get_projectors(N):
	P = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
	pauli_indx = get_indx(N)
	for n in range(4**N):
		to_append = np.array([1])
		for i in range(N):
			to_append = np.kron(to_append, P_single[pauli_indx[n][i]])
		P[n] = to_append @ to_append.conj().T
	return P

# Finding A
def get_A(P, sigma, rho, N):
	A = np.zeros((4**N, 4**N, 4**N, 4**N), dtype = np.complex128)
	A_unpacked = np.zeros((16**N, 16**N), dtype = np.complex128)
	for p in range(4**N):
		for q in range(4**N):
			for i in range(4**N):
				for j in range(4**N):
					temp = np.trace(P[q] @ sigma[i] @ rho[p] @ sigma[j].conj().T)
					A[p, q, i, j] = temp
					A_unpacked[(4**N) * p + q, (4**N) * i + j] = temp
	return A, A_unpacked

# Finding rho_final
def rhof_theoretical(U, rho, N):
	rho_final = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
	for i in range(4**N):
		rho_final[i] = U @ rho[i] @ U.conj().T
	return rho_final

# Finding P_pq given the projectors and final states
def get_probs(P, rho_final, N):
	Prob = np.zeros((4**N, 4**N), dtype = np.complex128)
	Prob_unpacked = np.zeros((16**N, ), dtype = np.complex128)
	for p in range(4**N):
		for q in range(4**N):
			temp = np.trace(P[q] @ rho_final[p])
			Prob[p, q] = temp
			Prob_unpacked[(4**N) * p + q] = temp
	return Prob, Prob_unpacked

# Given P_pq and A_pqij, find eta_ij
def get_eta(A_unpacked, Prob_unpacked, N):
	eta_unpacked = np.linalg.solve(A_unpacked, Prob_unpacked)
	eta = np.zeros((4**N, 4**N), dtype = np.complex128)
	for n in range(16**N):
		j = n % 4**N
		i = n // 4**N
		eta[i, j] = eta_unpacked[n]
	return eta

# Verification
def reconstruct(eta, sigma, rho, N):
	calc_rho = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
	for p in range(4**N):
		to_append = np.zeros((2**N, 2**N), dtype = np.complex128)
		for i in range(4**N):
			for j in range(4**N):
				to_append += eta[i, j] * sigma[i] @ rho[p] @ sigma[j].conj().T
		calc_rho[p] = to_append
	return calc_rho

# Single function to do everything
def find_eta(rho, phi, N):
	sigma = get_sigma(N)
	P = get_projectors(N)

	U = dynamics_unitary(phi, sigma, N)
	rho_final = rhof_theoretical(U, rho, N)

	_, Prob_unpacked = get_probs(P, rho_final, N)
	_, A_unpacked = get_A(P, sigma, rho, N)

	eta = get_eta(A_unpacked, Prob_unpacked, N)
	return eta, rho_final


if __name__ == "__main__":
	rho = get_rho(N)
	eta, rho_final = find_eta(rho, 1, N)

	sigma = get_sigma(N)
	calc_rho = reconstruct(eta, sigma, rho, N)

	# Output results
	print("eta_ij")
	print(np.round(eta, 5))
	print("\nLatex code for eta_ij")
	print_latex(np.round(eta, 5))
	print("\nSum of terms in (rho final - calculated rho)")
	print(np.sum(rho_final - calc_rho))
