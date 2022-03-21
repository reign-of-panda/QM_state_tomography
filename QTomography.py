import numpy as np
from copy import deepcopy
from scipy.linalg import sqrtm

import copy
######################################################################
# TODO
# - Angles from gaussian distribution
# - Fix state generation

######################################################################
# Finding eta

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
	return np.cos(phi) * np.identity(2**N) + 1j * np.sin(phi) * (sigma[3] / np.sqrt(2) + sigma[2] / np.sqrt(2))#2

def single_qubit(theta, phi):
	return np.array([[np.cos(theta/2)], [np.sin(theta/2) * np.exp(1j * phi)]])

# Initial states
# s0 = np.array([[1], [0]])
# s1 = np.array([[0], [1]])
# s2 = np.array([[1], [1]]) / np.sqrt(2)
# s3 = np.array([[1], [1j]]) / np.sqrt(2)
theta = 2 * np.pi / 3# 2 * np.arccos(1/np.sqrt(3))
s0 = single_qubit(0, 0)
s1 = single_qubit(theta, 0)
s2 = single_qubit(theta, 2 * np.pi / 3)
s3 = single_qubit(theta, -2 * np.pi / 3)
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

######################################################################
# Fidelities

def fidelity(A, B, pure = False):
	if not pure:
		return np.trace(sqrtm(sqrtm(A) @ B @ sqrtm(A)))**2
	else:
		return np.trace(A @ B)

def get_infidelities(A, B, N):
	infid = np.zeros((4**N, ))
	for i in range(4**N):
		infid[i] = 1 - np.real(fidelity(A[i], B[i], pure = False))
	return infid

def norm(eta1, eta2, n = 2):
	diff = eta1 - eta2
	if n == 1:
		return np.real(np.trace(sqrtm(diff.conj().T @ diff)))
	elif n == 2:
		return np.sqrt(np.real(np.trace(diff.conj().T @ diff)))

######################################################################
# State generation

def partial_trace(rho):
	s1 = np.array([[1], [0]])
	s2 = np.array([[0], [1]])
	s = np.array([s1, s2])
	tr = np.zeros((2, 2), dtype = np.complex128)
	for i in range(2):
		for j in range(2):
			for I in range(2):
				tr += s[i] @ s[j].conj().T * (np.kron(s[i], s[I]).conj().T @ rho @ np.kron(s[j], s[I]))
	return tr

CNOT = np.array([[1, 0, 0, 0],
 				 [0, 0, 0, 1],
 				 [0, 0, 1, 0],
 				 [0, 1, 0, 0]])

def u3(theta, phi, lam):
	c = np.cos(theta / 2)
	s = np.sin(theta / 2)
	e = np.exp(1j * phi)
	l = np.exp(1j * lam)
	return np.array([[c, -l * s], [e * s, l * e * c]])

def Cu3(theta, phi, lam):
	c = np.cos(theta / 2)
	s = np.sin(theta / 2)
	e = np.exp(1j * phi)
	l = np.exp(1j * lam)
	return np.array([[1, 0, 0, 0], 
					 [0, c, 0, -l*s],
					 [0, 0, 1, 0],
					 [0, e*s, 0, l*e*c]])

def mixed(rho, eps, args = None):
	r0 = np.zeros((2, 2), dtype = np.complex128)
	r0[0, 0] = 1

	rho2_i = np.kron(rho, r0)

	if args == None:
		theta = np.random.random() * np.pi * eps
		phi = np.random.random() * 2 * np.pi * eps
		lam = np.random.random() * 2 * np.pi * eps
	else:
		theta = args[0]
		phi = args[1]
		lam = args[2]
	rot_mat = u3(theta, phi, lam)
	H = CNOT @ np.kron(np.identity(2), rot_mat)

	rho2_f = H @ rho2_i @ H.conj().T

	mixed_rho = partial_trace(rho2_f)
	return mixed_rho

def mixed_general(rho, eps, args = None):
	r0 = np.zeros((2, 2), dtype = np.complex128)
	r0[0, 0] = 1

	rho2_i = np.kron(rho, r0)

	if args == None:
		# theta = np.random.uniform(0.5, 1) * np.pi * eps
		theta = np.arccos(np.random.uniform(-1, 1))
		phi = np.random.uniform() * 2 * np.pi * eps
		lam = np.random.uniform() * 2 * np.pi * eps
	else:
		theta = args[0]
		phi = args[1]
		lam = args[2]
	rot_mat = u3(theta, phi, lam)
	H = Cu3(theta, phi, lam) @ np.kron(np.identity(2), rot_mat)

	rho2_f = H @ rho2_i @ H.conj().T

	mixed_rho = partial_trace(rho2_f)
	return mixed_rho

def rho_experimental(eps, rho_t, N):
	rho_e = np.zeros(rho_t.shape, dtype = np.complex128)
	for i in range(rho_e.shape[0]):
		# theta = np.random.normal(scale = np.pi * eps)
		# phi = np.random.normal(scale = 2 * np.pi * eps)
		# lam = np.random.normal(scale = 2 * np.pi * eps)
		theta = np.arccos(np.random.uniform(-1, 1))
		phi = np.random.uniform(0, 2 * np.pi)
		lam = np.random.uniform(0, 2 * np.pi)
		rot_mat = u3(theta, phi, lam)
		rho_e[i] = rot_mat @ rho_t[i] @ rot_mat.conj().T
	return rho_e

def rho_exp_large_infid(eps, rho_t, N):
	temp_rho = copy.deepcopy(rho_t)
	for i in range(4**N):
		temp_rho[i] = pauli_1 @ temp_rho[i] @ pauli_1.conj().T
	rho_e = np.zeros(rho_t.shape, dtype = np.complex128)
	for i in range(rho_e.shape[0]):
		theta = np.random.normal(scale = np.pi * eps)
		phi = np.random.normal(scale = 2 * np.pi * eps)
		lam = np.random.normal(scale = 2 * np.pi * eps)
		rot_mat = u3(theta, phi, lam)
		rho_e[i] = rot_mat @ temp_rho[i] @ rot_mat.conj().T
	return rho_e

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
