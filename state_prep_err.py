import numpy as np
from QTomography import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


N = 2
phi = 1

eps = 1

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

def fidelity(A, B):
	return np.trace(sqrtm(sqrtm(A) @ B @ sqrtm(A)))**2

def get_infidelities(A, B, N):
	infid = np.zeros((4**N, ))
	for i in range(4**N):
		infid[i] = 1 - np.real(fidelity(A[i], B[i]))
	return infid

def single_trial(eps):
	rho_t = get_rho(N)
	rho_e = rho_experimental(eps, N)

	infidelities = get_infidelities(rho_t, rho_e, N)

	eta_t, _ = find_eta(rho_t, phi, N)
	eta_e, _ = find_eta(rho_e, phi, N)

	diff = eta_t - eta_e
	deta = np.real(np.trace(sqrtm(diff.conj().T @ diff)))
	# deta = np.sqrt(np.real(np.trace(diff.conj().T @ diff)))

	return np.mean(infidelities), deta

n_samples = 100
eps_max = 1
infids = np.zeros((n_samples, ))
detas = np.zeros((n_samples, ))
for i in range(n_samples):
	eps = i * eps_max / n_samples
	a, b = single_trial(eps)
	infids[i] = a
	detas[i] = b

data = np.column_stack((infids, detas))
np.savetxt("data.csv", data)

plt.title("State preparation errors")
plt.scatter(infids, detas)
plt.grid()
plt.xlabel("Infidelities")
plt.ylabel(r"$||\eta^{T} - \eta^{E}||$")
plt.show()
