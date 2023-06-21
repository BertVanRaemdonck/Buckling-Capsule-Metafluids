# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:00:57 2022

@author: u0123347
"""

import jsonTools as json
import interpolationTools as its

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import scipy.signal as sgn

plt.close('all')
plt.style.use(r'I:\PhD\Utilities\Python\adelStyle.mplstyle')

import matplotlib as mpl
mpl.use('Qt5Agg')

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

# --------------------------------
# %% Functions for processing data
# --------------------------------

def getPControlCurve(curve, mu=350e3, p0=p0):
    # get full PV curve
    v0 = curve['v']['infl'][0]
    pTotInfl = curve['p']['infl']*mu + p0*(v0/curve['v']['infl']-1)
    pTotDefl = curve['p']['defl']*mu + p0*(v0/curve['v']['defl']-1)

    # get inflation branch
    pkInd = sgn.find_peaks(curve['p']['infl'])[0][0]
    vInfl = curve['v']['infl'][:pkInd]
    pTotInfl = pTotInfl[:pkInd]

    # get deflation branch
        # throw away datapoints past the local minimum in the slope of the
        # remaining deflation branch
    vlInd = sgn.find_peaks(-pTotDefl)[0]
    if len(vlInd) <= 0:
        vlInd = np.argmin(-np.gradient(curve['p']['defl'], curve['v']['defl']))
    else:
        vlInd = vlInd[0]
    vDefl = curve['v']['defl'][:vlInd]
    pTotDefl = pTotDefl[:vlInd]

    vEndInfl = np.interp(pTotInfl[-1], np.flip(pTotDefl), np.flip(vDefl))
    vEndDefl = np.interp(pTotDefl[-1], pTotInfl, vInfl)

    return {'infl': {'v': np.append(vInfl, vEndInfl),
                     'p': np.append(pTotInfl, pTotInfl[-1])},
            'defl': {'v': np.append(vDefl, vEndDefl),
                     'p': np.append(pTotDefl, pTotDefl[-1])},
            'v0': v0}

def getVControlCurve(curve, mu=350e3, p0=p0):
    # get full PV curve
    v0 = curve['v']['infl'][0]
    pTotInfl = curve['p']['infl']*mu + p0*(v0/curve['v']['infl']-1)
    pTotDefl = curve['p']['defl']*mu + p0*(v0/curve['v']['defl']-1)

    # get inflation branch
    pkInd = sgn.find_peaks(curve['p']['infl'])[0][0]
    vInfl = curve['v']['infl'][:pkInd]
    pTotInfl = pTotInfl[:pkInd]

    # get deflation branch
    vlInd = np.argmax(np.gradient(curve['p']['defl'], curve['v']['defl']))
    pTotDefl = pTotDefl[:vlInd]
    vDefl = curve['v']['defl'][:vlInd]

    pEndInfl = np.interp(vInfl[-1], vDefl, pTotDefl)
    pEndDefl = np.interp(vDefl[-1], np.flip(vInfl), np.flip(pTotInfl))

    return {'infl': {'v': np.append(vInfl, vInfl[-1]),
                     'p': np.append(pTotInfl, pEndInfl)},
            'defl': {'v': np.append(vDefl, vDefl[-1]),
                     'p': np.append(pTotDefl, pEndDefl)},
            'v0': v0}

simDir = os.path.join(mainDir, '2.Numerics')
matchSimName = 'n=01_match_simulation.json'
matchSim = json.parseJson(os.path.join(simDir, r'10. ExperimentMatching\Final_experiments\Big_setup\Output', matchSimName))

# ---------------------------------
# %% Plot superposition explanation
# ---------------------------------

mu = 60e3
pvShell = getVControlCurve(matchSim, mu, 0.)
pvTot = getVControlCurve(matchSim, mu, p0)
pvTotP = getPControlCurve(matchSim, mu, p0)
v = np.linspace(pvTot['v0'], .1*pvTot['v0'], 200)
pg = p0*(pvShell['v0']/v-1)

plt.figure(figsize=(10,8))
for stroke in ('infl', 'defl'):
    plt.plot(1-pvShell[stroke]['v']/pvShell['v0'], 1e-3*pvShell[stroke]['p'],
             c='C0', label=r'shell $\Delta P_{shell}$' if stroke=='infl' else None)
    if stroke == 'infl':
        plt.plot(1-v/pvShell['v0'], 1e-3*pg, c='C2', label='gas $P_{int}$')
    plt.plot(1-pvTot[stroke]['v']/pvTot['v0'], 1e-3*pvTot[stroke]['p'],
             c='C4', label='total $P_{ext}$' if stroke=='infl' else None)
    plt.plot(1-pvTotP[stroke]['v']/pvTotP['v0'], 1e-3*pvTotP[stroke]['p'],
             c='C4', ls=':')
plt.legend(loc='upper right', framealpha=1.)
plt.xlabel(r'Volume $\Delta V/V_0$')
plt.ylabel(r'Pressure $P$ (kPa)')
plt.xlim(0, .4)
plt.ylim(0, 100.)
plt.tight_layout()
plt.show()
plt.savefig('Fig2Superposition.svg')

# -------------------------------------------------
# %% Plot simulation data for different stiffnesses
# -------------------------------------------------


plt.figure(figsize=(5,4.5))
for col, mu in zip(['#f3655c','C4'], [350e3, 60e3]):
    simData = getPControlCurve(matchSim, mu)
    plt.plot((1-simData['infl']['v']/simData['v0']), 1e-3*simData['infl']['p'],
             c=col, label='$G = $' + '{:.0f} kPa'.format(1e-3*mu))
    plt.plot((1-simData['defl']['v']/simData['v0']), 1e-3*simData['defl']['p'],
             c=col)
plt.xlim(0, .6)
plt.ylim(0, 200.)
plt.xlabel(r'$\Delta V/V_0$')
plt.ylabel(r'$P_{ext}$ (kPa)')
plt.legend(handlelength=1.25)
plt.tight_layout()
plt.show()
plt.savefig('Fig2InfluenceMaterial.svg')