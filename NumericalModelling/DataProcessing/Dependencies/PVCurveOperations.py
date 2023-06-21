# -*- coding: utf-8 -*-
"""
Functions for the representation and static
simulation of systems of elements with PV curves

Created on Thu Jan 28 14:58:56 2021

@author: Bert Van Raemdonck
@email: bert.vanraemdonck@kuleuven.be
"""

import numpy as np
import scipy.interpolate as itp
import plotTools as pts

class Branch:
    def __init__(self, v, p, parents=[]):
        self.v = np.array(v)
        self.p = np.array(p)
        self.getBounds()
        self.getPInterpolator()
        self.getVInterpolator()
        self.parents = []

    def getBounds(self):
        self.bounds = {'pmin': np.min(self.p), 
                       'pmax': np.max(self.p),
                       'vmin': np.min(self.v),
                       'vmax': np.max(self.v)}

    def getPInterpolator(self):
        mask = []
        for i, v in enumerate(self.v):
            if i<=1 or self.v[mask[0]] < self.v[mask[1]] < v or \
               self.v[mask[0]] > self.v[mask[1]] > v:
                mask.append(i)
        self.pInterpolator = itp.interp1d(self.v[mask], self.p[mask],
                                          fill_value='extrapolate')

    def getVInterpolator(self):
        mask = []
        for i, p in enumerate(self.p):
            if i<=1 or self.p[mask[0]] < self.p[mask[1]] < p or \
               self.p[mask[0]] > self.p[mask[1]] > p:
                mask.append(i)
        self.vInterpolator = itp.interp1d(self.p[mask], self.v[mask],
                                          fill_value='extrapolate')

    def getP(self, v, extrapolate=False):
        """
        Get pressure matching provided volume

        Interpolate the branch to get the pressure matching the provided
        volume value if that value is within the range of the data points.
        Otherwise, extrapolate the datapoints if extrapolate is set to True and
        don't return any matching if extrapolate is set to False

        Parameters
        ----------
        v : (array of) float(s)
            volume value(s) to get the matching pressure value at
        extrapolate : boolean
            select between extrapolating data and ingoring datapoints

        Returns
        -------
        list
            pressure values matching the provided volume values

        """
        # convert volume to array
        if not isinstance(v, (list, tuple, np.ndarray)):
            v = np.array([v])
        # initialize list of matching pressure values
        p = [None]*len(v)
        # interpolate all provided data points
        if self.pInterpolator is not None:
            for i, vPoint in enumerate(v):
                if extrapolate or \
                    self.bounds['vmax'] >= vPoint >= self.bounds['vmin']:
                    p[i] = self.pInterpolator(vPoint)

        return p

    def getV(self, p, extrapolate=False):
        """
        Get volume matching provided pressure

        Interpolate the branch to get the volume matching the provided
        pressure value if that value is within the range of the data points.
        Otherwise, extrapolate the datapoints if extrapolate is set to True and
        don't return any matching if extrapolate is set to False

        Parameters
        ----------
        p : (array of) float(s)
            pressure value(s) to get the matching volume value at
        extrapolate : boolean
            select between extrapolating data and ingoring datapoints

        Returns
        -------
        np.ndarray
            volume values matching the provided pressure values

        """
        # convert volume to array
        if not isinstance(p, (list, tuple, np.ndarray)):
            p = np.array([p])
        # initialize list of matching pressure values
        v = [None]*len(p)
        # interpolate all provided data points
        if self.vInterpolator is not None:
            for i, pPoint in enumerate(p):
                if extrapolate or \
                    self.bounds['pmax'] >= pPoint >= self.bounds['pmin']:
                    v[i] = self.vInterpolator(pPoint)

        return v

    def scaleV(self, vScale):
        """
        Scale volume axis

        Parameters
        ----------
        vScale : float
            Factor with which to scale the volume axis

        Returns
        -------
        Branch object
            New branch with scaled volume axis

        """
        return Branch(self.v*vScale, self.p)

    def addBranchV(self, *otherBranches):
        """
        Add branch objects along volume axis

        Parameters
        ----------
        *otherBranches : Branch object
            Branches of which to add the volume values to the volume values of
            this branch

        Returns
        -------
        Branch
            New branch with volumes added together

        """
        # Check if the branches have an overlapping interval in pressure
        branches = (self,) + otherBranches
        pMaxOverlap = np.min([branch.bounds['pmax'] for branch in branches])
        pMinOverlap = np.max([branch.bounds['pmin'] for branch in branches])
        
        if pMaxOverlap < pMinOverlap:
            return None
        else:
            # Create pressure levels in the overlapping range
            pLevels = np.array([])
            for branch in branches:
                indsInOverlap = (branch.p >= pMinOverlap) * \
                                (branch.p <= pMaxOverlap)
                pLevels = np.append(pLevels, branch.p[indsInOverlap])
            pLevels = np.unique(pLevels)
            # Add volumes at those pressure levels together
            vTotal = np.zeros(len(pLevels))
            for branch in branches:
                vTotal += branch.getV(pLevels)
            # Create new branch
            return Branch(vTotal, pLevels)

    @pts.createNewFigure(axInd=1, sizeInd=2, sizeDefault=(4.8,4.8))
    def plot(self, ax=None, size=None, *plotArgs, **plotKwargs):
        """
        Plot pressure-volume data of the branch on the provided axis

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis object
            Axis object to create the plot on
        size : (float, float)
            Default Axis size
        *plotArgs :
            arguments to pass to the plot function
        **plotKwargs :
            keyword arguments to pass to the plot function

        Returns
        -------
        None.

        """
        ax.plot(self.v, self.p, *plotArgs, **plotKwargs)


class PVCurve:
    def __init__(self, v=None, p=None):
        """
        Pressure-volume curve consisting of multiple branches
        
        This class provides operations on equilibrium PV curves like scaling
        them or adding them together, even if they are non-monotonic of consist
        of multiple branches. In order to initialize a PV curve with multiple
        disconnected equilibrium branches, refer to the addBranch function.

        Parameters
        ----------

        v : array, optional
            List of volume values of the consecutive points on the PV curve.
            The default is None.
        p : array, optional
            List of pressure values of the consecutive points on the PV curve.
            The default is None.

        Returns
        -------
        None.

        """
        self.branches = []
        if v is not None and p is not None:
            self.addBranch(Branch(v,p))


    def addBranch(self, *branch):
        """
        Add disconnected branch to the PV curve

        Parameters
        ----------
        branch : Branch object
            Branch to add to the PV curve

        Returns
        -------
        None.

        """
        for b in branch:
            self.branches.append(b)

    def curveToBranch(self):
        """
        Merge all curve branches into one

        Returns
        -------
        Branch object
            Branch containing data points of all the curve branches

        """
        vPoints = np.array([])
        pPoints = np.array([])
        for branch in self.branches:
            vPoints = np.append(vPoints, branch.v)
            pPoints = np.append(pPoints, branch.p)

        return Branch(vPoints, pPoints)


    def scaleV(self, vScale):
        """
        Scale volume axis

        Parameters
        ----------
        vScale : float
            Factor with which to scale the volume axis

        Returns
        -------
        PVCurve object
            New PV curve with scaled volume axis

        """
        scaledCurve = PVCurve()
        for branch in self.branches:
            scaledCurve.addBranch(branch.scaleV(vScale))
        return scaledCurve


    def scaleP(self, pScale):
        """
        Scale pressure axis

        Parameters
        ----------
        pScale : float
            Factor with which to scale the pressure axis

        Returns
        -------
        PVCurve object
            New PV curve with scaled pressure axis

        """
        scaledCurve = PVCurve()
        for branch in self.branches:
            scaledCurve.addBranch(branch.scaleP(pScale))
        return scaledCurve


    def multiply(self, N):
        """
        Calculate equilibrium configurations of multiple connected instances

        Return a new PV curve containing all equilibrium configurations in a
        system consisting of N instances of this PV curve. These points are in
        equilibrium in the sense that the pressure in every instance is the
        same and that the total volume is equal to the volume of the individual
        instances at that pressure. Not every point might be dynamically
        stable, however.

        Parameters
        ----------
        N : integer
            Amount of instances of this PV curve in a connected system

        Raises
        ------
        TypeError
            N can only be an integer

        Returns
        -------
        PVCurve object with branches representing all possible equilibrium
        configurations of N parent PV curves

        """
        if int(N) != N:
            raise TypeError('PVCurves can only be multiplied with integers')

        # Anonymous helper function
        def distributeAmounts(nbSlots, totalAmount):
            """
            Generate all possible distributions of positive integers over 
            nbSlots of which the total adds up to totalAmount and return it as
            a list of lists

            """
            if nbSlots == 1:
                # Trivial case: there is only one way to divide N over 1 slot
                return [[totalAmount,],]
            else:
                # Continue case: loop over all possible amounts to put in this
                # slot and use recursion to find all possible distributions
                # of the remaining amount over the remaining slots
                distribution = []
                for thisAmount in range(totalAmount+1):
                    otherAmounts = distributeAmounts(nbSlots-1, 
                                                      totalAmount-thisAmount)
                    for otherAmount in otherAmounts:
                        distribution.append([thisAmount,] + otherAmount)
                return distribution

        multipliedCurve = PVCurve()
        # Generate all unique possible distributions of N instances over the 
        # curve branches (e.g. 2 instances are on branch 1 and 3 instances are
        # on branch 2). We don't care about permutations because the curves
        # are the same for all instances
        branchDistributions = distributeAmounts(len(self.branches), N)
        # For every possible distribution, calculate the resulting branch
        for weights in branchDistributions:
            # Weigh every branch by the amount given by the distribution
            weightedBranches = []
            for i in range(len(weights)):
                if weights[i] > 0:
                    weightedBranch = self.branches[i].scaleV(weights[i])
                    weightedBranches.append(weightedBranch)
            # Add branches together
            weightedSum = weightedBranches[0].addBranchV(*weightedBranches[1:])
            weightedSum.parents = sum([[i,]*amount for i, amount in enumerate(weights)], start=[])
            multipliedCurve.addBranch(weightedSum)

        return multipliedCurve


    def addCurve(self, *otherCurves):
        """
        Calculate equilibrium configuration in a system with multiple curves

        Return a new PV curve containing all equilibrium configurations in a
        system consisting of one instance of this PV curve and one instance of
        every other supplied curve. These points are in equilibrium in the 
        sense that the pressure in every instance is the same and that the
        total volume is equal to the volume of the individual instances at that
        pressure. Not every point might be dynamically stable, however.

        Parameters
        ----------
        *otherCurves : PVCurve
            PV curves besides this one to add to the system for which
            equilibrium will be calculated

        Returns
        -------
        addedCurve : PVCurve
            PV curve with all equilibrium points in the system containing all
            parent curves

        """
        addedCurve = PVCurve()
        curves = (self,) + otherCurves
        
        # Calculate amount of possible combinations of branches
        nbBranches = [len(curve.branches) for curve in curves]
        nbCombinations = int(np.prod(nbBranches))
        # Indices of the branches for every curve that are considered in the
        # present combination
        selectedBranchInds = [0]*len(curves)

        # Generate all possible combinations
        for i in range(nbCombinations):
            # List of branch objects considered in the current combination
            selectedBranches = [0]*len(curves)
            for j,curve in enumerate(curves):
                selectedBranches[j] = curve.branches[selectedBranchInds[j]]
            # Add branches together
            combination = selectedBranches[0].addBranchV(*selectedBranches[1:])
            if combination is not None:
                combination.parents = selectedBranchInds[:]
                addedCurve.addBranch(combination)

            # Find next combination with different selected branch indices
            for j in range(len(curves)):
                selectedBranchInds[j] += 1
                if selectedBranchInds[j] >= len(curves[j].branches):
                    selectedBranchInds[j] = 0
                else:
                    break

        return addedCurve

    def getEnvelope(self, control='v', increasing=True, returnToStart=True):
        """
        Get the directional envelope of the PV-curve along an axis
        
        The envelope is obtained as follows:
            - start at a branch at the end of the curve
            - start increasing or decreasing a value for one of the axes
            - follow the branch until it reaches an extreme value along the
              controlled axis
            - if available, jump to the closest branch in a given direction
              along the uncontrolled axis
        For simple cases, this is equivalent to obtaining the dynamic curve
        from a static curve under the provided control

        Parameters
        ----------
        control : string, optional
            name of the controlled variable ('v' or 'p'). The default is 'v'.
        increasing : boolean, optional
            initially increase the controlled variable from the minimum to the
            maximum (True) or the other way around (False). The default is
            True.
        returnToStart : boolean, optional
            if True, reverse the direction on reaching the end of the initial
            increasing or decreasing phase. The default is True.

        Returns
        -------
        dynamicCurve : Branch object
            Single branch containing all envelope data

        """
        dynamicCurve = PVCurve()

        # Transform pressure and volume axis such that the following
        # calculations can always assume the case of volume control with
        # increasing volume
        branches = []
        flipFac = 1 if increasing else -1
        for branch in self.branches:
            if control.lower() in ('v', 'dv', 'volume'):
                branches.append(Branch(flipFac*branch.v, flipFac*branch.p))
            elif control.lower() in ('p', 'pressure'):
                branches.append(Branch(flipFac*branch.p, -flipFac*branch.v))

        # Start at the point with the lowest volume
        nextBranch = min(branches, key=lambda b: b.bounds['vmin'])
        vCurr = nextBranch.bounds['vmin']

        while nextBranch is not None:
            currBranch = nextBranch
            # Clip branch to start at the current end volume and sort points
            # in the right direction according to increasing/decreasing volume
            sortInds = np.argsort(currBranch.v)
            keepInds = [i for i in sortInds if currBranch.v[i] >= vCurr]
            # Add point to the branch corresponding to the projection from
            # the previous end volume with a slight offset to prevent double
            # values on the volume axis
            vCurr += 1e-6*(currBranch.bounds['vmax']-currBranch.bounds['vmin'])
            pProj = currBranch.getP(vCurr)[0]
            # Add clipped branch to the new curve
            if pProj is not None:
                newBranch = Branch(np.append(vCurr, currBranch.v[keepInds]), \
                                   np.append(pProj, currBranch.p[keepInds]))
                dynamicCurve.addBranch(newBranch)
            # Update current end point
            vCurr = currBranch.bounds['vmax']
            pCurr = currBranch.getP(vCurr)[0]
            # Find the branch with the highest pressure lower than at the end
            # of the current branch at the current volume
            nextBranch = None
            pBest = -np.inf
            for b in branches:
                if b.bounds['vmin'] <= vCurr <= b.bounds['vmax'] and \
                   b is not currBranch:
                    pBranch = b.getP(vCurr)[0]
                    if pBranch <= pCurr and pBranch > pBest:
                        nextBranch = b
                        pBest = pBranch

        # Apply inverse transformation on the pressure and volume axes to get
        # back to the original configuration and convert to a single Branch
        # object
        dynamicCurve = dynamicCurve.curveToBranch()
        if control.lower() in ('v', 'dv', 'volume'):
            dynamicCurve = Branch(dynamicCurve.v/flipFac, dynamicCurve.p/flipFac)
        elif control.lower() in ('p', 'pressure'):
            dynamicCurve = Branch(-dynamicCurve.p/flipFac, dynamicCurve.v/flipFac)

        if returnToStart:
            # Create branch in the other direction
            otherPart = self.getEnvelope(control, not increasing, False)
            # Merge both branches
            dynamicCurve = Branch(np.append(dynamicCurve.v, otherPart.v),
                                  np.append(dynamicCurve.p, otherPart.p))

        return dynamicCurve

    @pts.createNewFigure(axInd=1, sizeInd=2, sizeDefault=(4.8,4.8))
    def plot(self, ax=None, size=None, *plotArgs, **plotKwargs):
        """
        Plot all branches of the PV curve on the provided axis

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis object
            Axis object to create the plot on
        size : (float, float)
            default size of the Axis
        *plotArgs :
            arguments to pass to the plot function
        **plotKwargs :
            keyword arguments to pass to the plot function

        Returns
        -------
        None.

        """
        for branch in self.branches:
            branch.plot(ax, *plotArgs, **plotKwargs)
