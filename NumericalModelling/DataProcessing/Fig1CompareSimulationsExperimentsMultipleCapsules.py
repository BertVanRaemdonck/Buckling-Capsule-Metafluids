# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:53:33 2022

@author: u0123347
"""

import jsonTools as json
import interpolationTools as its
import PVCurveOperations as pvopts

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import scipy.optimize as opt
import re

plt.close('all')
plt.style.use(r'I:\PhD\Utilities\Python\matlabStyle.mplstyle')
colorPath = r'I:\PhD\Utilities\Python\colorPalette.json'
colorScheme = json.parseJson(colorPath)
curveCols = {ct: [colorScheme[ct][c] for c in ('verydark', 'dark', 'bright', 'light', 'verylight')] for ct in ('teal', 'orange')}
saveData = True

mainDir = r'G:\.shortcut-targets-by-id\1-3FB-HxWAHfT3_OfAz_xZChbL-kyzmYW\2.Collab_Metafluids'

# -------------------
# %% Shell parameters
# -------------------

muGreen = 350e3

ri = 8e-3
mu = muGreen

p0 = 101325
V0 = 4/3*np.pi*ri**3

# ------------------------
# %% Get experimental data
# ------------------------

expDir = os.path.join(mainDir, '5.Experiments')
expResDir = os.path.join(expDir, r'Final_experiments\Big_setup\Figure_1\Fig_1_b\data\data_figure')

expRaw = {}
expCorrected = {}

for fileName in os.listdir(expResDir):
    # parse filenames
    fileParams = re.findall(r'(.*)N=(\d+)_combination=A_(.+)_iteration=1.txt', fileName)
    if fileParams and ('corrected' in fileParams[0][0] or
                       fileParams[0][0] == ''):
        print(fileName)
        dataType, N, direction = fileParams[0]
        N = int(N)
        # get data from files
        expData = json.TDF(os.path.join(expResDir, fileName), delimiter=',')
        expPV = {'Dv': 1e-6*[v for k,v in expData.data.items() if 'volume' in k.lower()][0],
                 'p': 1e5*[v for k,v in expData.data.items() if 'pressure' in k.lower()][0]}
        if 'corrected' in dataType:
            if N not in expCorrected.keys():
                expCorrected[N] = {}
            expCorrected[N][direction] = expPV
        else:
            if N not in expRaw.keys():
                expRaw[N] = {}
            expRaw[N][direction] = expPV

plt.figure()
for i,N in enumerate(sorted(expRaw.keys())):
    for stroke in expRaw[N].values():
        plt.plot(stroke['Dv'], stroke['p'], 'C{}'.format(i))
plt.show()


# --------------------------------
# %% Get data on system compliance
# --------------------------------

expSysDir = os.path.join(expDir, r'Final_experiments\Big_setup\Figure_1\Fig_1_a\Match_sim_exp')
sysResults = json.TDF(os.path.join(expSysDir, 'system_characterization (2).txt'), delimiter=',')
sysData = {'Dv': 1e-6*sysResults['# volume (mL)'],
           'p': 1e5*sysResults[' pressure (Bar)']}

vThreshold = .4e-6
pFit = sysData['p'][sysData['Dv'] > vThreshold]
vFit = sysData['Dv'][sysData['Dv'] > vThreshold]

coeffs = np.polyfit(vFit, pFit, 2)
vOffset = opt.fsolve(lambda v: np.polyval(coeffs, v), vThreshold)
pSysData = np.polyval(coeffs, sysData['Dv']+vOffset)
vSystem = itp.interp1d(pSysData, sysData['Dv'], kind='linear', fill_value='extrapolate')

# # reconstruct function that interpolates the system compliance
#     # get monotonically increasing top envelope
# monoIndsTop = np.full(sysData['p'].shape, True, dtype=bool)
# highestP = sysData['p'][0]
# for i, p in enumerate(sysData['p'][1:]):
#     monoIndsTop[i+1] = p > highestP
#     if monoIndsTop[i+1]:
#         highestP = p
# pSystemTop = itp.interp1d(sysData['Dv'][monoIndsTop], sysData['p'][monoIndsTop],
#                           kind='linear', fill_value='extrapolate')
#     # get monotonically increasing bottom envelope
# monoIndsBot = np.full(sysData['p'].shape, True, dtype=bool)
# lowestP = sysData['p'][-1]
# for i, p in reversed(list(enumerate(sysData['p'][:-1]))):
#     monoIndsBot[i] = p < lowestP
#     if monoIndsBot[i]:
#         lowestP = p
# pSystemBot = itp.interp1d(sysData['Dv'][monoIndsBot], sysData['p'][monoIndsBot],
#                           kind='linear', fill_value='extrapolate')

# pSysMono = .5*(pSystemTop(sysData['Dv']) + pSystemBot(sysData['Dv']))
# vSystem = itp.interp1d(pSysMono, sysData['Dv'], kind='linear', fill_value='extrapolate')


# -----------------------------------------
# %% Get simulation data for a single shell
# -----------------------------------------

simDir = os.path.join(mainDir, '2.Numerics')
matchSimName = 'n=01_match_simulation.json'
matchSim = json.parseJson(os.path.join(simDir, r'10. ExperimentMatching\Final_experiments\Big_setup\Output', matchSimName))
for strokeName, stroke in zip(('forward', 'backward'), ('infl', 'defl')):
    vi = (ri/matchSim['params']['geometry']['ri'])**3*matchSim['v'][stroke]
    matchSim[strokeName] = {'Dv': V0 - vi,
                            'p': mu*matchSim['p'][stroke] + p0*(V0/vi - 1)}

PVSim = pvopts.PVCurve()
for stroke in (matchSim['forward'], matchSim['backward']):
    for i in range(1, len(stroke['Dv'])-1):
        if stroke['p'][i] >= max(stroke['p'][i-1:i+2]) or \
           stroke['p'][i] <= min(stroke['p'][i-1:i+2]):
            break
    PVSim.addBranch(pvopts.Branch(stroke['Dv'][:i], stroke['p'][:i]))


# --------------------------------------
# %% Get simulations for multiple shells
# --------------------------------------

pSysData = np.linspace(np.min(matchSim['forward']['p']), np.max(matchSim['forward']['p']), 500)
vSysData = vSystem(pSysData)
PVSys = pvopts.PVCurve(vSysData, pSysData)

simCorrected = {}
simRaw = {}
for N in list(expCorrected.keys()) + [50, 100, 200]:
    PVSimMult = PVSim.multiply(N)
    simCorrected[N] = PVSimMult.getEnvelope()
    simRaw[N] = PVSimMult.addCurve(PVSys).getEnvelope()

    if saveData:
        for dataName, data in zip(('raw', 'compliance'), (simCorrected[N], simRaw[N])):
            splitInd = np.argmax(data.v)
            for stroke, strokeInds in zip(('forward', 'backward'),
                                          (slice(0,splitInd), slice(splitInd+1,len(data.v)))):
                fileName = 'sim{}_N={}_combination=A_{}_iteration=1.txt'.format(dataName, N, stroke)
                print(fileName)
                with open(fileName, 'w') as outFile:
                    outFile.write('# Volume (mL), Pressure (Bar)')
                    for p, v in zip(data.p[strokeInds], data.v[strokeInds]):
                        outFile.write('\n{:e},{:e}'.format(1e6*v, 1e-5*p))

# ----------------------------------------------
# %% Get pressure drops for every buckling event
# ----------------------------------------------

drops = {}
for N, curve in simRaw.items():
    drops[N] = []
    for i in range(len(curve.p)-1):
        Dp = curve.p[i] - curve.p[i+1]
        if curve.v[i+1] > curve.v[i] and Dp > 0:
            drops[N].append(Dp)

import json as json2

plt.figure()
for N, Dps in drops.items():
    plt.scatter(np.full(len(Dps),N), Dps, 30, 'C0')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.xlabel('$N$')
plt.ylabel('$\Delta p$ [Pa]')
plt.show()

with open('CompareSimulationsExperimentsFig1b-simulatedDrops.json', 'w') as outFile:
    json2.dump(drops, outFile, indent=4)


# --------------------
# %% Compare PV curves
# --------------------

plt.figure()
for i, N in enumerate(sorted(expRaw.keys())):
    for strokeName, stroke in expRaw[N].items():
        plt.plot(1e6*stroke['Dv'], 1e-5*stroke['p'], curveCols['teal'][::-1][i],
                 label='N={}, exp'.format(N) if strokeName=='forward' else None)
for i, N in enumerate(sorted(expRaw.keys())):
    plt.plot(1e6*simRaw[N].v, 1e-5*simRaw[N].p, c=curveCols['orange'][::-1][i],
             lw=2, label='N={}, sim'.format(N))
plt.xlim(left=0, right=40)
plt.ylim(bottom=0, top=2.25)
plt.xlabel(r'volume $\Delta V$ [mL]')
plt.ylabel(r'pressure $p$ [bar]')
plt.legend(ncol=2, handlelength=1., loc='upper left', framealpha=1.)
plt.title('raw experimental data, simulations with compliance')
plt.show()

if saveData:
    plt.savefig('CompareSimulationsExperimentsFig1b_raw.pdf')


plt.figure()
for i, N in enumerate(sorted(expCorrected.keys())):
    for strokeName, stroke in expCorrected[N].items():
        plt.plot(1e6*stroke['Dv']/N, 1e-5*stroke['p'], curveCols['teal'][::-1][i],
                 label='N={}, exp'.format(N) if strokeName=='forward' else None)
for i, N in enumerate(sorted(expCorrected.keys())):
    plt.plot(1e6*simCorrected[N].v/N, 1e-5*simCorrected[N].p, c=curveCols['orange'][::-1][i],
             lw=2, label='N={}, sim'.format(N))
plt.xlim(left=0, right=1.5)
plt.ylim(bottom=0, top=2.25)
plt.xlabel(r'volume $\Delta V$ [mL]')
plt.ylabel(r'pressure $p$ [bar]')
plt.legend(ncol=2, handlelength=1., loc='upper left', framealpha=1.)
plt.title('experimental data corrected, simulations without compliance')
plt.show()

if saveData:
    plt.savefig('CompareSimulationsExperimentsFig1b_normalized.pdf')