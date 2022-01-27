import numpy as np
from QTomography import *

rho = get_rho(1)

a = np.array([[0.8], [0.6j]])
rho[3] = a @ a.conj().T

pauli_0 = np.identity(2)
pauli_1 = np.array([[0, 1], [1, 0]])
pauli_2 = np.array([[0, -1j], [1j, 0]])
pauli_3 = np.array([[1, 0], [0, -1]])
pauli = np.array([pauli_0, pauli_1, pauli_2, pauli_3])

B = np.zeros((4, 4, 4), dtype = np.complex128)
for p in range(4):
	for i in range(4):
		for j in range(4):
			temp = np.trace(pauli[i] @ rho[p] @ pauli[j].conj().T)
			temp = np.round(temp, 5)
			B[p, i, j] = temp
print(B)
print()
print(np.sum(B, axis = 0))