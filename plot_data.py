import numpy as np
import matplotlib.pyplot as plt

file = "data.csv"
infids, detas = np.loadtxt(file, unpack = True)

grads = detas[20:]**2 / infids[20:]

m_min = np.min(grads)
m_max = np.max(grads)

print(f"Lower gradient: {m_min}")
print(f"Upper gradient: {m_max}")
print(f"Upper / Lower: {m_max / m_min}")

lower = lambda x: m_min * x
upper = lambda x: m_max * x

max_infid = np.max(infids)
max_deta2 = np.max(detas)**2
x = np.linspace(0, max_infid * 1.1, 30)

plt.title("State preparation errors (N = 1)")
plt.scatter(infids, detas**2, marker = "x")
plt.fill_between(x, lower(x), upper(x), color = "lightblue", alpha = 0.5)
plt.grid()
plt.xlabel("Infidelities")
plt.ylabel(r"$||\eta^{T} - \eta^{E}||^{2}_{1}$")
plt.xlim([-max_infid * 0.05, max_infid * 1.1])
plt.ylim([-max_deta2 * 0.05, max_deta2 * 1.1])
plt.show()