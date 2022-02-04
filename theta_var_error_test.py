# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:13:48 2022

@author: therm
"""

"""
Plotting the 3D plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



file_name = "Andres_3M_runs.csv"

def lin(x):
    m = 0.175/0.069
    c = 0
    return m *x + c


all_dF = pd.read_csv(file_name)
infidelity = all_dF['infidelity']

lin_diff = lin(infidelity) # Subtract from the data to see how it affects the surface plot

err_sq = all_dF['error']**2 #- lin_diff


# plt.plot(infidelity, err_sq, ',', alpha = 1, color = 'blue') # i[:-16]
# plt.grid()
# plt.minorticks_on()
# plt.xlabel('Infidelities')
# plt.ylabel("$||\eta^T - \eta^E ||^{2}_1$")
# plt.title("State Preparation Errors ($3 \\times 10^6$ samples)")
# plt.xlim(0, 0.015)
# plt.ylim(0, 0.035)
# plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
# plt.show()

# Pick the mask based on whether you want theo whole range or a limited range of data
# mask = (err_sq <= 0.05) & (infidelity <= 0.02)
mask = [True] * len(infidelity)

plot_type = "scatter"

i_lim = infidelity[mask]
e_lim = err_sq[mask]
hist, xedges, yedges = np.histogram2d(i_lim, e_lim, bins=100)
# Keep bins to less than 100 if you have many data points

midx = abs(xedges[1] - xedges[0]) / 2
midy = abs(yedges[1] - yedges[0]) / 2

xpos, ypos = np.meshgrid(xedges[:-1] + midx, yedges[:-1] + midy, indexing="ij")
zs = np.array(np.ravel(hist))
Z = zs.reshape(xpos.shape)


fig = plt.figure()
ax = plt.axes(projection='3d')
spacing = 15

if plot_type == "bar":
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    
    # Construct arrays with the dimensions for the 16 bars.
    dx = midx * np.ones_like(zpos) * 2
    dy = midy * np.ones_like(zpos) * 2
    dz = hist.ravel()

    my_cmap = plt.get_cmap("viridis")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color = my_cmap(rescale(dz)))

elif plot_type == "scatter":
    ax.scatter(xpos, ypos, hist, c=hist, cmap='coolwarm', linewidth=0.5);

elif plot_type == "trisurf":
    ax.plot_trisurf(xpos.flatten(), ypos.flatten(), hist.flatten(), linewidth=0.2, antialiased=True, cmap = 'coolwarm')

elif plot_type == "surface":
    ax.plot_surface(xpos, ypos, Z, cmap = 'coolwarm')

ax.set_xlabel("Infidelity", labelpad = spacing)
ax.set_ylabel("$||\eta^T - \eta^E ||^{2}_1$", labelpad = spacing)
ax.set_zlabel("Frequency", labelpad = spacing)
plt.show()


"""
These lines are used for combining multiple csv files into a single file
"""
# path = os.getcwd()
# all_files = os.listdir(path)

# csv_files = []
# for i in all_files:
#     if i[-3:] == 'csv':
#         csv_files.append(i)
   
# # print(csv_files)
# # #%%
# all_dF = [[], []]
# for i in csv_files:
#     if i[0:4] == 'N1S6':
#         print(i)
#         dF = np.loadtxt(path + "\\" + i)
#         dF = dF.T
#         dF = dF.tolist()
#         infid = dF[0]
#         err = dF[1]
#         all_dF[0] += infid
#         all_dF[1] += err

# new_dF = np.array(all_dF).T
# new_dF2 = pd.DataFrame(new_dF, columns = ["infidelity", "error"])
# new_dF2.to_csv('Andres_3M_runs.csv')

# np.savetxt("million_runs.csv", new_dF)



