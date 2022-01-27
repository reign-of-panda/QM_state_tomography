import numpy as np
from QTomography import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


N = 1
phi = 1

# eps = 0.01

def rho_experimental(eps, N):
	rho = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
	pauli_indx = get_indx(N)
	for n in range(4**N):
		to_append = np.array([1])
		for i in range(N):
			rand_real = np.random.random((2, 1))
			rand_imag = np.random.random((2, 1))

			new_state = s[pauli_indx[n][i]] + eps * (rand_real + 1j * rand_imag)
			new_state = new_state / np.linalg.norm(new_state)

			to_append = np.kron(to_append, new_state)
		rho[n] = to_append @ to_append.conj().T
	return rho

def fidelity(A, B, pure = False):
	if not pure:
		return np.trace(sqrtm(sqrtm(A) @ B @ sqrtm(A)))**2
	else:
		return np.trace(A @ B)

def get_infidelities(A, B, N):
	infid = np.zeros((4**N, ))
	for i in range(4**N):
		infid[i] = 1 - np.real(fidelity(A[i], B[i], pure = True))
	return infid

def single_trial(eps, N):
	rho_t = get_rho(N)
	rho_e = rho_experimental(eps, N)

	infidelities = get_infidelities(rho_t, rho_e, N)

	eta_t, _ = find_eta(rho_t, phi, N)

	# Find eta_e
	sigma = get_sigma(N)
	P = get_projectors(N)

	U = dynamics_unitary(phi, sigma, N)
	rho_final = rhof_theoretical(U, rho_e, N)

	_, Prob_unpacked = get_probs(P, rho_final, N)
	_, A_unpacked = get_A(P, sigma, rho_t, N)

	eta_e = get_eta(A_unpacked, Prob_unpacked, N)

	# Find RHS
	RHS = np.zeros((4**N,), dtype = np.complex128)
	for p in range(4**N):
		for i in range(4**N):
			for j in range(4**N):
				for I in range(4**N):
					for J in range(4**N):
						RHS[p] += eta_e[i, j] * eta_t[I, J] * np.trace(sigma[I] @ rho_t[p] @ sigma[J].conj().T @ sigma[i] @ rho_t[p] @ sigma[j].conj().T)

	return 1 - infidelities, RHS

a, b = single_trial(0.1, N)
print("Different p")
print(a)
print(np.real(b))
print("Sum")
print(np.sum(a))
print(np.sum(np.real(b)))
