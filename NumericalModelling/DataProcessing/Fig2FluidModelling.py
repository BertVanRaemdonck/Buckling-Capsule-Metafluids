# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:38:04 2023

@author: bertv
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cmx
import mpl_toolkits.axes_grid1.inset_locator as inset
import scipy.signal as sgn
import scipy.optimize as opt
import scipy.interpolate as itp
import scipy.signal as sgn
import sys
import os

sys.path.append(r'I:\PhD\Metafluids\Python\Utilities')
import SphericalShell as sph
import interpolationTools as its
import parameterStudy as prm
import PVCurveOperations as pvopts
import jsonTools as json

plt.close('all')
plt.style.use(r'I:\PhD\Utilities\Python\adelStyleSmall.mplstyle')

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5.5,5.5))

colSph = 'C0'
colCol = 'C4'

cNorm  = mcol.Normalize(vmin=0, vmax=1)
cm = plt.get_cmap('viridis')
cm = mcol.ListedColormap([[0.267, 0.224, 0.514, 1.0]])
cm = mcol.ListedColormap([[0.712, 0.702, 0.812, 1.0]])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

# ------------
# %% Load data
# ------------

# Load data
mainDir = r'G:\Mijn Drive'
#simDir = os.path.join(mainDir, '2.Numerics')
simDir = mainDir
matchSimName = 'n=01_match_simulation.json'
matchSim = json.parseJson(os.path.join(simDir, r'10. ExperimentMatching\Final_experiments\Big_setup\Output', matchSimName))

# Process data
simCurve = sph.SphericalShell(matchSim, R=9e-3, G=350e3)


# -----------------------------------------------------
# %% Plot system curves for different numbers of shells
# -----------------------------------------------------

for N, ax in zip((1,2,10), (axs[0,0], axs[0,1], axs[1,0])):
    # calculate system curve
    sysCurve = simCurve.staticCurve.multiply(N)
    # plot static branches
    for j, branch in enumerate(sysCurve.branches):
        nbCollapsed = np.count_nonzero(branch.parents)
        ax.plot(1e6*branch.v, 1e-3*branch.p,
                c=scalarMap.to_rgba(.9*nbCollapsed/N), #alpha=.4,
                label=f'$N$ = {N}' if j==0 else None)
    # plot dynamic curve
    dynCurve = sysCurve.getEnvelope(control='v', returnToStart=False)
    ax.plot(1e6*dynCurve.v, 1e-3*dynCurve.p, 'C0', lw=1.8)
    # format axis
    ax.set_xlim((0,1.1*N))
    ax.set_ylim((0,175))
    ax.set_xlabel('Volume $\Delta V$ (ml)')
    ax.set_ylabel('Pressure $P$ (kPa)')
    ax.legend(loc='lower right', handlelength=1)


# ------------------------
# %% Pressure drop scaling
# ------------------------

ax = axs[1,1]
dropValsN = []
dropValsP = []

# calculate pressure drops
for N in np.hstack((np.logspace(0,2,20), [1e2,])):
    N = int(N)
    print(f'\rcalculating drops for N = {N}         ', end='')
    dropValsN.append(N)
    staticCurve = simCurve.staticCurve.multiply(N)
    loadingCurve = staticCurve.getEnvelope(control='v', returnToStart=False)
    loadingDrops = np.diff(loadingCurve.p)[1:]
    dropValsP.append(-loadingDrops[np.where(loadingDrops < 0)])
print()

for i, (dropsN, dropsP) in enumerate(zip(dropValsN, dropValsP)):
    ax.scatter(np.full(dropsP.shape, dropsN), 1e-3*dropsP, 2, 'C0')

# plot evolution of drops with the number of shells
topCoords = [1e-3*p[0] for p in dropValsP]
bottomCoords = [1e-3*p[-1] for p in dropValsP]
ax.add_patch(plt.Polygon(np.vstack((np.hstack((dropValsN, np.flip(dropValsN))),
                                    np.hstack((topCoords, np.flip(bottomCoords))))).T,
                         ec=None, alpha=.4, label='Simulation'))

# # analytical formula for large number of shells
# for i, branch in enumerate(simCurve.staticCurve.branches[:2]):
#     # get slope of the collapsed branch at the critical pressure of one capsule
#     # (evaluate it at a slightly lower pressure than the critical one because
#     # the slope changes rapidly in the neighborhood of the critical point due
#     # to dynamic effects)
#     slope = np.gradient(branch.p, branch.v)
#     slopeCrit = its.interp1d(.98*simCurve.snapData['p']['l']['s']['pExt'],
#                              branch.p, slope)
#     # get the change in volume associated with the snapping of one capsule
#     # under pressure control
#     dvCrit = simCurve.snapData['p']['l']['d']['Dv']
#     # the drop in pressure is approximately equal to the product of DvCrit and
#     # the slope of the system curve, which is approximately slopeCrit/N
#     dropValsPAn = dvCrit*slopeCrit/np.array(dropValsN)
#     ax.plot(dropValsN, 1e-3*dropValsPAn, 'C0--', label='Analytical' if i==0 else None)

# format axis
ax.set_xlabel('Number of shells $N$ (-)')
ax.set_ylabel('Pressure drop during \n collapse $\Delta P_{drop}$ (kPa)')
#ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim((.8,120))
ax.set_ylim((0,60))
#ax.legend(handlelength=1, frameon=False)


# ----------------
# %% Format figure
# ----------------

fig.tight_layout()
for ax, label in zip(axs.flatten(), ('A', 'B', 'C', 'D')):
    pos = ax.get_position()
    fig.text(pos.xmin-.11, pos.ymax+.007, r'\textbf{\Large '+label+'}')

fig.show()
