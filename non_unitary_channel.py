import numpy as np
from QTomography import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import time

N = 1
phi = 1

# Checked Hermitian, trace preservation, positive semi-definite
def random_eta(trace_preserving = True):
	eig_vals = np.random.random(4)
	if trace_preserving:
		eig_vals = eig_vals / np.sum(eig_vals)

	phi = 2 * np.pi * np.random.random(2 * 4**N)
	theta = np.arccos(2 * (np.random.random(2 * 4**N) - 0.5)) # Uniform in cos(theta)

	eta = np.zeros((4**N, 4**N), dtype = np.complex128)
	for i in range(0, 2 * 4**N, 2):
		vec1 = single_qubit(theta[i], phi[i]).reshape(2, 1)
		vec2 = single_qubit(theta[i + 1], phi[i + 1]).reshape(2, 1)
		vec = np.kron(vec1, vec2)

		eta += eig_vals[i // 2] * (vec @ vec.conj().T)

	return eta

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
		infid[i] = 1 - np.real(fidelity(A[i], B[i]))
	return infid

eta = random_eta()
print(eta)
quit()
print()
def single_trial(eps, N):
	sigma = get_sigma(N)

	rho_ti = get_rho(N)
	mat = np.zeros((4, 4), dtype = np.complex128)
	for i in range(4):
		for j in range(4):
			mat[i, j] = eta[i, j] * np.trace(sigma[i] @ rho_ti[0] @ sigma[j].conj().T)
	print(mat)
	quit()
	rho_tf = reconstruct(eta, sigma, rho_ti, N)
	for i in range(rho_tf.shape[0]):
		print(np.sum((rho_tf[i] - rho_tf[i].conj().T)**2))
		print(np.trace(rho_tf[i]))
		print(np.linalg.eigvals(rho_tf[i]))
		print()

	rho_ei = rho_experimental(eps, rho_ti, N)
	rho_ef = reconstruct(eta, sigma, rho_tf, N)

	print(get_infidelities(rho_ti, rho_ei, N))
	print(get_infidelities(rho_tf, rho_ef, N))

single_trial(0.01, 1)
quit()

n_samples = 1000000
eps_max = 0.05
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

