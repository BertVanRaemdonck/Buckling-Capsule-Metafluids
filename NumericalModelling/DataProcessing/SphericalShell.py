# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:52:08 2022

@author: u0123347
"""

import numpy as np
import scipy.interpolate as itp
import scipy.signal as sgn
import matplotlib.pyplot as plt
import copy

import PVCurveOperations as pvOpts
import interpolationTools as its


class SphericalShell:

    def __init__(self, curve, G=1., R=1., p0=101325., pOver=0,
                 getStaticCurve=True, deflVelThr=10., deflMinThr=.25,
                 nbInterpPts=None):
        """
        Characteristic of a capsule based on simulation data

        This class post-processes dimensionless simulation data to take into
        account the influence of the material stiffness, the shell size and the
        internal pressure (assuming isothermal ideal gas behavior) on the
        external pressure of the shell. It also separates the data in spherical
        and collapsed branches and extracts the critical points.

        Properties of interest of this class:

            - SphericalShell.dynamicData:
                dictionary holding the processed simulation data before
                splitting into a spherical and collapsed branch. The keys are
                - 't': simulation time
                - 'Dp': differential pressure across the capsule shell
                - 'pExt': total external pressure of the capsule
                - 'pInt': internal pressure inside the capsule cavity
                - 'Dv': change in capsule volume from the stress-free state
                - 'vExt': external volume occupied by the capsule
                - 'vInt': internal volume of the capsule cavity
                - 'A': area of the inner surface in self-contact (if available)
                - 'Ax': x-coordinate of the center of contact (if available)
                - 'Ay': y-coordinate of the center of contact (if available)
                - 'Dy': inwards axial displacement of the capsule pole
                the values are dictionaries with as keys 'infl' and 'defl' and
                as values the values of the corresponding variables throughout
                the loading and unloading stroke, respectively, of the
                simulation. All values are in SI units.

            - SphericalShell.staticBranches:
                array with the processed data after splitting into a spherical
                and collapsed branch. The first element corresponds to the
                spherical branch and the second to the data for the collapsed
                branch. Both elements are dictionaries with the same keys as
                the dynamicData dictionary and as values the data for those
                variables on the respective branches. The spherical branch
                corresponds to the inflation stroke of the simulation up to
                the critical point and the collapsed branch corresponds to the
                deflation stroke of the simulation up to the critical point
                under volume control.

            - SphericalShell.staticCurve:
                PVCurve object (see package PVCurveOperations) featuring the
                spherical and the collapsed branch split into monotonically
                increasing or decreasing segments. These branches can be
                accessed using staticCurve.branches[i], where usually
                    i=0 is the spherical branch
                    i=1 is the collapsed branch up to the minimum in pressure
                    i=2 is the collapsed branch after that minimum
                In theory, this object allows to easily simulate the
                interaction between this curve and other PV curves. However,
                these simulations do not correctly take into account the
                non-monotonicity of the collapsed branch and can be influenced
                by numerical noise, so caution is advised. Usually, the most
                robust way to obtain reasonably accurate simulations is to
                create a new PVCurve object only featuring the branches for
                i=0 and i=1.

            - SphericalShell.snapData:
                dictionary with the data associated with the critical points.
                It is a nested dictionary in which each level corresponds to
                a certain property of the snapping event. The keys for each
                level are:
                    1. 'p': the instability occurs under pressure control
                       'v': the instability occurs under volume control
                    2. 'l': the instability occurs under loading (V drops)
                       'u': the instability occurs under unloading (V rises)
                    3. 's': get the value of the variable at the start point of
                            the instability
                       'e': get the value of the variable at the end point of
                            the instability
                       'd': get the difference in the value of the variable
                            between the start and the end of the instability.
                            this is a shortcut for e-s
                For example: the internal volume of the capsule at which the
                instability starts under pressure controlled unloading is given
                by snapData['p']['s']['vInt'].
                Finally, this dictionary contains the dissipated energy on a
                loading-unloading cycle for pressure and volume control by
                selecting 'DU' as key on the second level.

        Parameters
        ----------
        curve : dictionary
            simulation data for the desired shell geometry extracted from a
            json report file that contains both the simulation results as the
            simulation parameters
        G : float, optional
            shell material shear modulus in Pa. The default is 1.0 which
            leaves the differential shell pressure dimensionless
        R : float, optional
            midplane radius of the shell in m. The default is 1.0.
        p0 : float, optional
            atmospheric pressure in Pa. The default is 101325. Set to 0 to
            exclude the effect of the internal pressure on the characteristic
        pOver : float, optional
            pressure relative to atmosphere inside the cavity of the shell when
            the shell is in the stress-free configuration. The default is 0.
        getStaticCurve : boolean, optional
            whether or not to generate the staticCurve property. The default
            is True.
        deflVelThr : float, optional
            the critical point on deflation is determined as the point where
            the velocity of the pole exceeds deflVelThr times the average
            velocity in the quasi-static regime. Lower values result in
            earlier detection. Values that are too high might result in no
            detection at all, in which case the critical point is found as
            the point where the velocity is maximal. The default is 10..
        deflMinThr : float, optional
            the critical point on pressure controlled unloading is found as
            the first local minimum on unloading that is the minimum of the
            8 closest points and does not lie above the fraction deflMinThr
            of all points on the collapsed branch. Putting this fraction to 1.
            always selects the global minimum on the collapsed branch.
            Decreasing it allows to find points that are not the global minimum
            but that are more correct because the global minimum can be caused
            by numerical noise due to contact. The default is .25.

        """
        self.origData = curve
        self.params = curve['params']

        self.G = G
        self.R = R
        self.p0 = p0
        self.pOver = pOver
        self.v0 = self.R**3 * self.params['geometry']['V0']
        self.eta = self.params['geometry']['eta']
        self.delta = self.params['imperfection']['delta']

        self.deflVelThr = deflVelThr
        self.deflMinThr = deflMinThr

        self.interpData = self.interpolateData(nbInterpPts) if nbInterpPts is not None else self.origData
        self.dynamicData = self.applyScale(self.G, self.R, self.p0)

        self.staticBranches, self.velQS = self.getQuasistaticBranches()
        self.staticCurve = pvOpts.PVCurve()
        if getStaticCurve:
            if self.staticBranches is not None:
                for branch in self.staticBranches:
                    startInd = 0
                    endInd = 0
                    while endInd < len(branch['Dv'])-1:
                        endInd = its.getMonotonicSegmentEnd(branch['pExt'], startInd)
                        if endInd-startInd > 2:
                            self.staticCurve.addBranch(pvOpts.Branch(branch['Dv'][startInd:endInd+1],
                                                                     branch['pExt'][startInd:endInd+1]))
                            startInd = endInd
                        else:
                            endInd = len(branch['Dv'])
                # for branch in self.staticBranches:
                #     self.staticCurve.addBranch(pvOpts.Branch(branch['Dv'], branch['Dp']))
            else:
                self.staticCurve.addBranch(pvOpts.Branch(self.dynamicData['Dv']['infl'],
                                                         self.dynamicData['Dp']['infl']))

            self.snapData = self.getCriticalInds()
            self.snapData = self.extendSnapData(self.snapData)
            self.pCrUp = self.snapData['p']['l']['s']['pExt']
            self.pCrDown = self.snapData['p']['u']['s']['pExt']

    def interpolateData(self, nbPts):
        interpData = {}
        for name, data in self.origData.items():
            if isinstance(data, dict) and 'infl' in data.keys() and 'defl' in data.keys():
                interpData[name] = {}
                for stroke in ('infl', 'defl'):
                    # generate sample points based on the indices. This makes
                    # sure that the non-uniform time stepping in the simulation
                    # is preserved such that the resolution is higher in the
                    # neighborhood of the snapping points
                    normPtsOrig = np.linspace(0, 1, len(data[stroke]))
                    normPtsNew = np.linspace(0, 1, nbPts)
                    # interpolate data at these sample points
                    interpData[name][stroke] = itp.interp1d(normPtsOrig, data[stroke], kind='quadratic')(normPtsNew)

        return interpData


    def applyScale(self, G, R, p0):
        scaledData = {k: {} for k in ('t', 'Dp', 'pExt', 'pInt', 'Dv', 'vExt', 'vInt', 'A', 'Ax', 'Ay', 'Dy')}

        for stroke in ('infl', 'defl'):
            # apply scaling laws
            scaledData['t'][stroke] = self.interpData['t'][stroke]
            scaledData['Dp'][stroke] = G * self.interpData['p'][stroke]
            scaledData['vInt'][stroke] = R**3 * self.interpData['v'][stroke]
            scaledData['Dy'][stroke] = R * self.interpData['d'][stroke]
            if 'A' in self.origData.keys():
                scaledData['A'][stroke] = R**2 * self.interpData['A'][stroke]
                scaledData['Ax'][stroke] = R * self.interpData['Ax'][stroke]
                scaledData['Ay'][stroke] = R * self.interpData['Ay'][stroke]
            else:
                scaledData['A'][stroke] = np.full(self.interpData['t'][stroke].shape, np.nan)
                scaledData['Ax'][stroke] = np.full(self.interpData['t'][stroke].shape, np.nan)
                scaledData['Ay'][stroke] = np.full(self.interpData['t'][stroke].shape, np.nan)

            # derived volume measures
            vRubber = 4./3.*np.pi*self.R**3*((1+.5*self.eta)**3-(1-.5*self.eta)**3)
            scaledData['vExt'][stroke] = scaledData['vInt'][stroke] + vRubber
            vInt0 = scaledData['vInt']['infl'][0]
            scaledData['Dv'][stroke] = vInt0 - scaledData['vInt'][stroke]

            # derived pressure measures
            scaledData['pInt'][stroke] = (self.p0+self.pOver)*self.v0 / scaledData['vInt'][stroke] - self.p0
            scaledData['pExt'][stroke] = scaledData['pInt'][stroke] + scaledData['Dp'][stroke]

        return scaledData


    def getQuasistaticBranches(self):
        debug = False
        if self.eta == .01 and round(self.params['imperfection']['eccentricity'],3) == 0.:
            debug = False#True

        # get point at which the inflation reaches maximum pressure locally
        critIndInfl = np.where(self.dynamicData['Dp']['infl'][:-1] > \
                               self.dynamicData['Dp']['infl'][1:])[0]
        if len(critIndInfl) > 0:
            critIndInfl = critIndInfl[0]
        else:
            return None, None
        vCrit = self.dynamicData['Dv']['infl'][critIndInfl]
        
        # the snapping transition on deflation occurs between this critical
        # point and the point where the pressure reaches a local maximum on
        # unloading. Also disregard the last two points of the defl stroke
        # because there can be a lot of noise on these
        if debug:
            plt.figure()
            for stroke in ('defl',):
                plt.plot(self.dynamicData['Dv'][stroke], 
                         self.dynamicData['Dp'][stroke], 'r:')

        unloadMask = np.where(self.dynamicData['Dv']['defl'] < vCrit)[0][:-1]
        unloadMask = np.hstack([unloadMask[0]-1, unloadMask])
        pkInds = sgn.find_peaks(self.dynamicData['Dp']['defl'][unloadMask])[0]
        if len(pkInds) > 0:
            highestPeakInd = pkInds[np.argmax(self.dynamicData['Dp']['defl'][unloadMask][pkInds])]
            unloadMask = unloadMask[:highestPeakInd+1]


        # calculate average velocity of the pole in the quasi-static part of
        # the deflation before unbuckling. This is the part with volume
        # greater than the critical volume on inflation
        tDefl = self.dynamicData['t']['defl']
        velDefl = np.gradient(self.dynamicData['Dy']['defl'], tDefl)
        velDeflCrit = np.interp(vCrit, np.flip(self.dynamicData['Dv']['defl']), np.flip(velDefl))
        maskDefl = np.where(np.logical_and(self.dynamicData['Dv']['defl'] > vCrit,
                                            np.abs(velDefl) > .01*abs(velDeflCrit)))
        # abs(self.dynamicData['Dy']['defl']) < .95*abs(self.dynamicData['Dy']['defl'][0]))) #
        velDeflQS = np.trapz(np.abs(velDefl[maskDefl]), x=tDefl[maskDefl]) / np.ptp(tDefl[maskDefl])

        # the snapping transition on deflation happens when the pole velocity
        # becomes significantly larger than the average velocity of the pole
        # in the quasi-static regime
        critIndDefl = np.where(np.abs(velDefl[unloadMask]) > self.deflVelThr*velDeflQS)[0]
        if len(critIndDefl) > 0:
            critIndDefl = critIndDefl[0]
            weight = 1 - (np.abs(velDefl[unloadMask][critIndDefl]) - self.deflVelThr*velDeflQS) / \
                          (np.abs(velDefl[unloadMask][critIndDefl]) - np.abs(velDefl[unloadMask][critIndDefl-1]))
        else:
            #pInflItp = itp.interp1d(self.dynamicData['Dv']['infl'], self.dynamicData['Dp']['infl'], fill_value='extrapolate')
            #pOnInfl = pInflItp(self.dynamicData['Dv']['defl'])
            critIndDefl = np.argmin(np.gradient(self.dynamicData['Dp']['defl'],
                                                self.dynamicData['Dv']['defl'])[unloadMask])
            weight = 1
        critIndDefl = unloadMask[critIndDefl]
            
        # pInflItp = itp.interp1d(self.dynamicData['Dv']['infl'], self.dynamicData['Dp']['infl'], fill_value='extrapolate')
        # pOnInfl = pInflItp(self.dynamicData['Dv']['defl'])
        # pDiff = pOnInfl - self.dynamicData['Dv']['defl']
        # dpDiffdt = np.gradient(pDiff, self.dynamicData['t']['defl'])
        # critIndDefl = unloadMask[np.argmin(dpDiffdt[unloadMask])]
        # weight = 1
        # velDeflQS = 0.
        
        if debug:
            plt.scatter(self.dynamicData['Dv']['defl'][critIndDefl],
                        self.dynamicData['Dp']['defl'][critIndDefl], 30, 'r')


        # clip both inflation and deflation branches until the critical points
        branches = []
        for stroke, critInd in zip(('infl', 'defl'), (critIndInfl, critIndDefl)):
            branchData = {}
            for k, v in self.dynamicData.items():
                valCrit = (1-weight)*v[stroke][critInd-1] + weight*v[stroke][critInd]
                branchData[k] = np.append(v[stroke][:critInd], valCrit)
            branches.append(branchData)
        
        if debug:
            for branch in branches:
                plt.plot(branch['Dv'], branch['Dp'], 'k')

        return branches, abs(velDeflQS)


    def getData(self, branchID, indOnBranch):
        if self.staticBranches is not None:
            return {k: v[indOnBranch] for k, v in self.staticBranches[branchID].items()}
        else:
            return {k: np.nan for k in self.dynamicData.keys()}


    def getCriticalInds(self):
        if self.staticBranches is not None and \
           all([len(b)>2 for b in self.staticBranches]):
            # critical points at the ends of the quasi-static branches
            ipLoading = np.argmax(self.staticBranches[0]['pExt'])
            ivLoading = np.argmax(self.staticBranches[0]['Dv'])
            ivUnloading = np.argmin(self.staticBranches[1]['Dv'])

            # critical pressure on inflation is the local minimum, but make
            # it robust against numerical noise
            pDefl = self.staticBranches[1]['pExt']
            vDefl = self.staticBranches[1]['Dv']
            # find the 8 closest points to the data points in volume
            minWindow = 8
            for ipUnloading in range(len(pDefl)-1,-1,-1):
                iLeft = ipUnloading
                iRight = ipUnloading
                while iRight - iLeft < minWindow-1:
                    if iRight >= len(pDefl)-1:
                        iLeft -= 1
                    elif iLeft <= 0:
                        iRight += 1
                    elif abs(vDefl[iLeft-1]-vDefl[ipUnloading]) < \
                         abs(vDefl[iRight+1]-vDefl[ipUnloading]):
                        iLeft -= 1
                    else:
                        iRight += 1
                # it is the minimum of these 8 points
                if pDefl[ipUnloading] <= min(pDefl[iLeft:iRight+1]):
                    # it is not overruled by a broad lower minimum
                    nbLowerPoints = len(pDefl[pDefl<pDefl[ipUnloading]])
                    if nbLowerPoints < self.deflMinThr*len(pDefl):
                        break
        else:
            ipLoading = np.nan
            ipUnloading = np.nan
            ivLoading = np.nan
            ivUnloading = np.nan

        return {'p': {'l': {'s': {'i': ipLoading}}, 'u': {'s': {'i': ipUnloading}}},
                'v': {'l': {'s': {'i': ivLoading}}, 'u': {'s': {'i': ivUnloading}}}}



    def extendSnapData(self, snapData):
        extendedData = copy.deepcopy(snapData)
        if self.staticBranches is not None:            
            for control, controlData in extendedData.items():
                critVar = 'pExt' if control == 'p' else 'Dv'
                for stroke, startBranch, endBranch in zip(('l', 'u'), (0,1), (1,0)):
                    strokeData = controlData[stroke]

                    # get value of all parameters at the start of the instability
                    startData = strokeData['s']
                    snapInd = startData['i']
                    for k, v in self.getData(startBranch, snapInd).items():
                        startData[k] = v

                    # get value of all parameters at the end of the instability
                    strokeData['e'] = {}
                    inds = np.arange(len(self.staticBranches[endBranch]['pExt']))
                    endInd = its.interpAll(startData[critVar],
                                           self.staticBranches[endBranch][critVar],
                                           inds, nMax=1)
                    # print(startData[critVar])
                    # print(self.staticBranches[endBranch][critVar])
                    # print(inds)
                    # print(endInd)
                    # print()
                    endInd = endInd[0] if len(endInd) > 0 else 0
                    for k, branchData in self.staticBranches[endBranch].items():
                        if not np.isnan(branchData[0]):
                            strokeData['e'][k] = its.interp1d(endInd, inds, branchData)
                        else:
                            strokeData['e'][k] = np.nan

                    # get difference between start and end of the instability for
                    # all parameters
                    strokeData['d'] = {}
                    for k in self.staticBranches[0].keys():
                        strokeData['d'][k] = strokeData['e'][k] - strokeData['s'][k]

                # get dissipated energy on loading and unloading
                    # 1) point at the end of the instability on unloading
                dynP = np.array([extendedData[control]['u']['e']['pExt'],])
                dynV = np.array([extendedData[control]['u']['e']['Dv'],])
                    # 2) spherical branch until instability on unloading
                startInd = np.where(self.staticBranches[0]['Dv'] >= dynV[-1])[0][0]
                endInd = extendedData[control]['l']['s']['i']
                dynP = np.hstack((dynP, self.staticBranches[0]['pExt'][startInd:endInd+1]))
                dynV = np.hstack((dynV, self.staticBranches[0]['Dv'][startInd:endInd+1]))
                    # 3) point at the end of the instability on unloading
                dynP = np.append(dynP, extendedData[control]['l']['e']['pExt'])
                dynV = np.append(dynV, extendedData[control]['l']['e']['Dv'])
                    # 4) collapsed branch until instability on loading
                startInd = np.where(self.staticBranches[1]['Dv'] <= dynV[-1])[0][0]
                endInd = extendedData[control]['u']['s']['i']
                dynP = np.hstack((dynP, self.staticBranches[1]['pExt'][startInd:endInd+1]))
                dynV = np.hstack((dynV, self.staticBranches[1]['Dv'][startInd:endInd+1]))
                    # 5) close loop
                dynP = np.append(dynP, dynP[0])
                dynV = np.append(dynV, dynV[0])
                    # calculate pdv integral along this path
                extendedData[control]['DU'] = np.trapz(dynP, x=dynV)

        else:
            extendedData = {}
            for control in ('p', 'v'):
                extendedData[control] = {}
                for stroke in ('l', 'u'):
                    extendedData[control][stroke] = {}
                    for point in ('s', 'e', 'd'):
                        extendedData[control][stroke][point] = {}
                        for var in self.dynamicData.keys():
                            extendedData[control][stroke][point][var] = np.nan

        return extendedData


    def plot(self):
        fig, ax = plt.subplots()
        
        for stroke in ('infl', 'defl'):
            ax.plot(self.dynamicData['Dv'][stroke]/self.v0,
                     self.dynamicData['pExt'][stroke], 'C0')

        for control in ('p', 'v'):
            for stroke in ('l', 'u'):
                plt.plot([self.snapData[control][stroke]['s']['Dv']/self.v0,
                          self.snapData[control][stroke]['e']['Dv']/self.v0],
                         [self.snapData[control][stroke]['s']['pExt'],
                          self.snapData[control][stroke]['e']['pExt']], 'o--')

        ax.set_xlabel(r'Volume $\Delta V/V_0$ (-)')
        ax.set_ylabel(r'Pressure $P_{ext}$ (-)')
        fig.show()


    def plotCurve(self, ax, scaleP=1., scaleV=None, labelFun=None, **plotOpts):
        if scaleV is None:
            scaleV = 1/self.v0
        if labelFun is None:
            labelFun = lambda c: None
        # plot static branches
        if self.staticBranches is not None:
            for j, branch in enumerate(self.staticBranches):
                contactMask = np.where(branch['A'] < 1e-2)[0]
                notContactMask = np.where(branch['A'] >= 1e-2)[0]
                for mask, ls, toLabel in zip((contactMask, notContactMask), ('-','--'), (True,False)):
                    ax.plot(scaleV*branch['Dv'][mask],
                            scaleP*branch['pExt'][mask], ls=ls, **plotOpts,
                            label=labelFun(self) if j==0 and toLabel else None)
        # plot snapping transitions under volume control
        for lu in ('l', 'u'):
            p = [scaleP*self.snapData['v'][lu][se]['pExt'] for se in ('s', 'e')]
            v = [scaleV*self.snapData['v'][lu]['s']['Dv'],]*2
            ax.plot(v, p, ls=':', **plotOpts)