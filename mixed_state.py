import numpy as np
from QTomography import *
from partial_trace import partial_trace

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

def mixed(rho, eps):
	r0 = np.zeros((2, 2), dtype = np.complex128)
	r0[0, 0] = 1

	rho2_i = np.kron(rho, r0)

	theta = np.random.random() * np.pi * eps
	phi = np.random.random() * 2 * np.pi * eps
	lam = np.random.random() * 2 * np.pi * eps
	rot_mat = u3(theta, phi, lam)
	H = CNOT @ np.kron(np.identity(2), rot_mat)

	rho2_f = H @ rho2_i @ H.conj().T

	mixed_rho = partial_trace(rho2_f)
	return mixed_rho

if __name__ == "__main__":
	r0 = np.zeros((2, 2), dtype = np.complex128)
	r0[0, 0] = 1
	mr0 = mixed(r0, 0.1)
	print(mr0)
	print(np.trace(mr0))
	print(np.trace(mr0 @ mr0))


