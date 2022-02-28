import numpy as np
from QTomography import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import time

N = 1
phi = 1

eps = 0.1

def u3(theta, phi, lam):
	c = np.cos(theta / 2)
	s = np.sin(theta / 2)
	e = np.exp(1j * phi)
	l = np.exp(1j * lam)
	return np.array([[c, -l * s], [e * s, l * e * c]])

def rho_experimental(eps, rho_t, N):
	rho_e = np.zeros(rho_t.shape, dtype = np.complex128)
	for i in range(rho_e.shape[0]):
		theta = np.random.random() * np.pi * eps
		phi = np.random.random() * 2 * np.pi * eps
		lam = np.random.random() * 2 * np.pi * eps
		rot_mat = u3(theta, phi, lam)
		rho_e[i] = rot_mat @ rho_t[i] @ rot_mat.conj().T
	return rho_e

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
	rho_e = rho_experimental(eps, rho_t, N)
	# for i in range(len(rho_e)):
	# 	print(np.trace(rho_e[i]))
	# 	print(np.trace(rho_e[i] @ rho_e[i]))
	# 	print(np.sum(abs(rho_e[i] - rho_e[i].conj().T)**2))

	infidelities = get_infidelities(rho_t, rho_e, N)

	eta_t, _ = find_eta(rho_t, phi, N)
	sigma = get_sigma(N)
	print(np.round(eta_t, 5))
	print(np.trace(reconstruct(eta_t, sigma, rho_t, N)[1]))
	quit()

	# Find eta_e
	sigma = get_sigma(N)
	P = get_projectors(N)

	U = dynamics_unitary(phi, sigma, N)
	rho_final = rhof_theoretical(U, rho_e, N)

	_, Prob_unpacked = get_probs(P, rho_final, N)
	_, A_unpacked = get_A(P, sigma, rho_t, N)

	eta_e = get_eta(A_unpacked, Prob_unpacked, N)

	diff = eta_t - eta_e
	# deta = np.real(np.trace(sqrtm(diff.conj().T @ diff)))
	deta = np.sqrt(np.real(np.trace(diff.conj().T @ diff)))

	return np.mean(infidelities), deta

n_samples = 1000000
eps_max = 0.5
infids = np.zeros((n_samples, ))
detas = np.zeros((n_samples, ))
start = time.time()
for i in range(n_samples):
	if i%100 == 0 and i != 0:
		current = time.time()
		print(f"\tIteration: {i}\tCurrent time (mins): {np.round((current - start) / 60, 2)}\tRemaining time: {np.round((n_samples * (current - start) / i - (current - start)) / 60, 2)}")
	eps = i * eps_max / n_samples
	a, b = single_trial(eps_max, N)
	infids[i] = a
	detas[i] = b

data = np.column_stack((infids, detas))
np.savetxt("oasihd2.csv", data)

plt.title("State preparation errors")
plt.plot(infids, detas**2, ",")
plt.grid()
plt.xlabel("Infidelities")
plt.ylabel(r"$||\eta^{T} - \eta^{E}||^{2}_{1}$")
plt.show()

