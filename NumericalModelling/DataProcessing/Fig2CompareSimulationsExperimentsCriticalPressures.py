# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:10:14 2022

@author: u0123347
"""

import parameterStudy as prm
import jsonTools as json
import interpolationTools as its

import os
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
plt.style.use(r'I:\PhD\Utilities\Python\matlabStyle.mplstyle')

mu = 350e3
Ro = 10e-3
p0 = 101325

saveData = False

# -------------------------
# %% Load experimental data
# -------------------------

expFolder = r'G:\.shortcut-targets-by-id\1-3FB-HxWAHfT3_OfAz_xZChbL-kyzmYW\2.Collab_Metafluids\5.Experiments\Final_experiments\Big_setup\Figure_2\Fig2.a\output\data'

expData = {}
for stroke in ('forward', 'backward'):
    fileName = '{}_critical_pressures.csv'.format(stroke)
    fileData = json.TDF(os.path.join(expFolder, fileName), delimiter=',')
    expData[stroke] = fileData.data

# -----------------------
# %% Load simulation data
# -----------------------

simFolder = r'I:\PhD\Metafluids\Abaqus\MasterIndex\Results\MatchExperimentTrends'
simStudy = prm.ParameterStudy(simFolder, ('eta',), (lambda c: c['params']['geometry']['eta'],))

# calculate total pressure for the given material and inner pressure
plt.figure()
for i, curve in enumerate(simStudy.flatResults):
    V0 = curve['params']['geometry']['V0']
    curve['ptot'] = {}
    for stroke in ('infl', 'defl'):
        ps = curve['p'][stroke]
        vi = curve['v'][stroke]
        pg = p0*(V0/vi-1)
        curve['ptot'][stroke] = mu*ps + pg
        plt.plot(vi, curve['ptot'][stroke], 'C{}'.format(i))
plt.show()

# get critical pressure
simData= {}
for stroke, stepName, getPeak in zip(('forward', 'backward'), ('infl', 'defl'), (its.interpMax, its.interpMin)):
    simData[stroke] = {'d/R': np.array([]), 'Critical pressure': np.array([])}
    for curve in sorted(simStudy.flatResults, key=lambda c: c['params']['geometry']['eta']):
        simData[stroke]['d/R'] = np.append(simData[stroke]['d/R'], curve['params']['geometry']['eta'])
        _, pCrit = getPeak(curve['v'][stepName][10:], curve['ptot'][stepName][10:], mode='first')
        simData[stroke]['Critical pressure'] = np.append(simData[stroke]['Critical pressure'], pCrit if pCrit is not None else np.nan)


# write data to csv files
if saveData:
    # buckling pressure files
    for strokeName, strokeData in simData.items():
        fileName = '{}_critical_pressures.csv'.format(strokeName)
        with open(fileName, 'w') as outFile:
            outFile.write('d/R, Iteration, Critical pressure')
            for eta, p in zip(strokeData['d/R'], strokeData['Critical pressure']):
                outFile.write('\n{},1,{}'.format(eta, 1e-5*p))

# ------------------------
# %% Get analytical trends
# ------------------------

Ri = lambda eta: Ro*(2-eta)/(2+eta)
V0 = lambda eta: 4/3.*np.pi*Ri(eta)**3
Vc = lambda eta: V0(eta) * (1-eta*(1-eta))
pgc = lambda eta: 0*p0*(V0(eta)/Vc(eta)-1)

pfAn = lambda eta: 4*mu*eta**2 + pgc(eta)
pbAn = lambda eta: 3*mu*eta**2.5 + pgc(eta)

# ------------------
# %% Compare results
# ------------------

plt.figure()
for i, stroke in enumerate(('forward', 'backward')):
    plt.scatter(expData[stroke]['d/R'], expData[stroke]['Critical pressure'],
                30, 'C{}'.format(i), label='{} (exp.)'.format(stroke))
    plt.plot(simData[stroke]['d/R'], 1e-5*simData[stroke]['Critical pressure'],
             'C{}d--'.format(i), label='{} (sim.)'.format(stroke))
#plt.plot(expData['forward']['d/R'], 1e-5*pfAn(expData['forward']['d/R']), 'C0:')
#plt.plot(expData['forward']['d/R'], 1e-5*pbAn(expData['forward']['d/R']), 'C1:')
plt.legend(ncol=2)
plt.xlabel(r'thickness $d/R$ [-]')
plt.ylabel('pressure $p$ [bar]')
plt.show()

