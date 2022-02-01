import numpy as np
import matplotlib.pyplot as plt

infids = np.array([0.])
detas = np.array([0.])
for i in range(3):
	infids_, detas_ = np.loadtxt(f"N1S6_{i}.csv", unpack = True)

	infids = np.append(infids, infids_)
	detas = np.append(detas, detas_**2)

dx = 0.0001
dy = 0.0002

x = 0.010
y = np.linspace(0.013, 0.040, 40)
freqs = np.zeros(y.shape)

indx = np.where((infids >= x - dx/2) & (infids <= x + dx/2), True, False)
cross_y = detas[indx]

for i in range(len(y)):
	print(i)
	indy = np.where((cross_y >= y[i] - dy/2) & (cross_y <= y[i] + dy/2), 1, 0)
	freqs[i] = np.sum(indy)

plt.title(f"Cross-section at an infidelity of {x}")
plt.plot(y, freqs / (dx * dy))
plt.grid()
plt.xlabel(r"$||\eta^{T} - \eta^{E}||^{2}_{1}$")
plt.ylabel("Frequency denisty")
plt.show()