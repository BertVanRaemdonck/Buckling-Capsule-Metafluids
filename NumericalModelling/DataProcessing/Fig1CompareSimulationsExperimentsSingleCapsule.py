# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:53:33 2022

@author: u0123347
"""

import jsonTools as json
import interpolationTools as its

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as itp

plt.close('all')
plt.style.use(r'I:\PhD\Utilities\Python\matlabStyle.mplstyle')
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
expResDir = os.path.join(expDir, r'Final_experiments\Big_setup\Figure_1\Fig_1_a\Match_sim_exp')

expData = {}
for stroke in ('forward', 'backward'):
    expPath = os.path.join(expResDir, 'pv{}_0.txt'.format(stroke[0]))
    expResults = json.TDF(expPath, delimiter=',')
    expData[stroke] = {'Dv': 1e-6*expResults['# Volume injected (mL)'], 
                       'p': 1e5*expResults['pressure (bar)']}

sysPath = os.path.join(expResDir, 'system_characterization (2).txt')
sysResults = json.TDF(sysPath, delimiter=',')
sysData = {'Dv': 1e-6*sysResults['# volume (mL)'],
           'p': 1e5*sysResults[' pressure (Bar)']}

plt.figure()
for stroke in expData.values():
    plt.plot(stroke['Dv'], stroke['p'])
plt.plot(sysData['Dv'], sysData['p'])
plt.show()

# ----------------------------------------------------------------------
# %% Simulate effect of system compliance on simulation after correction
# ----------------------------------------------------------------------

# get true system compliance curve
# this corrects for the fact that in the compliance measurement, in the first
# part there is a realignment of the syringe barrel which distorts the
# measurements. Therefore, the compliance curve is based on the measurement
# past this point
vThreshold = .4e-6
pFit = sysData['p'][sysData['Dv'] > vThreshold]
vFit = sysData['Dv'][sysData['Dv'] > vThreshold]

coeffs = np.polyfit(vFit, pFit, 2)
vOffset = opt.fsolve(lambda v: np.polyval(coeffs, v), vThreshold)
pSysData = np.polyval(coeffs, sysData['Dv']+vOffset)
vSystem = itp.interp1d(pSysData, sysData['Dv'], kind='linear', fill_value='extrapolate')

plt.figure()
plt.plot(sysData['Dv'], sysData['p'])
plt.plot(sysData['Dv'], pSysData)
plt.show()

if saveData:
    with open('system_characterization_corrected.txt', 'w') as outFile:
        outFile.write('# volume (mL), pressure (Bar)')
        for p,v in zip(pSysData, sysData['Dv']):
            outFile.write('\n{},{}'.format(1e6*v,1e-5*p))

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


# take a PV curve simulated without system compliance and approximate its
# behavior if it were put in a system with compliance
def simulateWithCompliance(p, vPerfect, vSystemFun):
    vTot = vPerfect + vSystemFun(p)
    vPrev = vTot[0]
    vSimCorr = [vPerfect[0]]
    pSimCorr = [p[0]]
    i = 1
    while i < len(vTot):
        if its.liesBetween(vPrev, vTot[0], vTot[i], include1=True, include2=True):
            vSimCorr.append(vTot[i])
            pSimCorr.append(p[i])
            vPrev = vTot[i]
            i+=1
        else:
            # skip data points
            while not its.liesBetween(vPrev, vTot[0], vTot[i], include1=True, include2=True) and i < len(vTot):
                i += 1
            # add a new data point by interpolation at the same volume as the
            # last valid data point to get perfectly straight snaps
            vSimCorr.append(vPrev)
            pSimCorr.append(p[i-1] + (p[i]-p[i-1])/(vTot[i]-vTot[i-1])*(vPrev-vTot[i-1]))
    return np.array(vSimCorr), np.array(pSimCorr)


# ----------------------
# %% Get simulation data
# ----------------------

simDir = os.path.join(mainDir, '2.Numerics')
matchSimName = 'n=01_match_simulation.json'
matchSim = json.parseJson(os.path.join(simDir, r'10. ExperimentMatching\Final_experiments\Big_setup\Output', matchSimName))
for strokeName, stroke in zip(('forward', 'backward'), ('infl', 'defl')):
    vi = (ri/matchSim['params']['geometry']['ri'])**3*matchSim['v'][stroke]
    matchSim[strokeName] = {'Dv': V0 - vi,
                            'p': mu*matchSim['p'][stroke] + p0*(V0/vi - 1)}
    vCorr, pCorr = simulateWithCompliance(matchSim[strokeName]['p'],
                                          matchSim[strokeName]['Dv'],
                                          vSystem)
    matchSim[strokeName]['DvCompl'] = vCorr
    matchSim[strokeName]['pCompl'] = pCorr

if saveData:
    for stroke in ('forward', 'backward'):
        for name, pLabel, vLabel in zip(('sim_raw', 'sim_compliance'),
                                        ('p', 'pCompl'), ('Dv', 'DvCompl')):
            fileName = 'pv{}_0_{}.txt'.format(stroke[0], name)
            with open(fileName, 'w') as outFile:
                outFile.write('# Volume injected (mL), pressure (bar)')
                for p, v in zip(matchSim[stroke][pLabel], matchSim[stroke][vLabel]):
                    outFile.write('\n{:e},{:e}'.format(1e6*v, 1e-5*p))


# --------------------
# %% Compare PV curves
# --------------------

plt.figure()
plt.title('Single shell with $t=2$ mm, $r_o=10$ mm, Green')
# plot average experimental curve
for stroke in expData.keys():
    plt.plot(1e6*expData[stroke]['Dv'],
             1e-5*expData[stroke]['p'],
             'C0', label='experiment' if stroke=='forward' else None)
# plot simulated curve
for stroke in ('forward', 'backward'):
    plt.plot(1e6*matchSim[stroke]['DvCompl'],
              1e-5*matchSim[stroke]['pCompl'],
              'C1', label=r'simulation ($\mu$ = {:.3f} MPa, $\delta$ = {})'.format(1e-6*mu, matchSim['params']['imperfection']['delta']) if stroke=='forward' else None)

# format axes
plt.xlabel(r'$\Delta V$ [ml]')
plt.ylabel(r'$p$ [bar]')
plt.xlim(left=0, right=2.5)
plt.ylim(bottom=0, top=2.)
plt.legend()

plt.show()

# if saveData:
#     plt.savefig('CompareSimulationsExperimentsFig1a.pdf')
