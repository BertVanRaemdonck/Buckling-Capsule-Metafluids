# -*- coding: utf-8 -*-
"""
Functions to interpolate series of data, built around numpy and scipy
- Interpolation2D class
    Robust interpolation of 2D surfaces in the presence of NaN values
- interp1d
    Point interpolation of 1D data of which both axes can be non-monotonic
- interpAll
    Find all points that interpolate to a value in a non-monotonic array
- interpMax, interpMin
    Interpolation to find locations of peaks more precisely than by index

Created on Tue Jun  8 09:38:03 2021
Last revision on Mon July 12 2021

@author: Bert Van Raemdonck
@email: bert.vanraemdonck@kuleuven.be
@correspondence: benjamin.gorissen@kuleuven.be

requires
    arrayTools.py (custom) to parse meshgrids

"""

import numpy as np
import scipy.interpolate as itp

import arrayTools as ats


class Interpolation2D:
    def __init__(self, xGrid, yGrid, zGrid, kind='linear'):
        """
        Smoothly interpolate 2D gridded data z = f(x,y)

        Interpolation is performed by first interpolating the x-values
        and then the y-values using the scipy.interpolate.interp1d function
        with the provided interpolation method. In case the data contains NaN
        values, NaN will also be interpolated there, but otherwise these
        entries do not affect the result (which they seem to do in the
        scipy.interpolate.interp2d function)
        
        The interpolation object can be evaluated by calling it with an x and
        y sample argument

        Parameters
        ----------
        xGrid : mxn np.ndarray
            x-coordinates of data points in meshgrid format
        yGrid : mxn np.ndarray
            y-coordinates of data points in meshgrid format
        zGrid : mxn np.ndarray
            matching z-coordinates of the data points
        kind : string, optional
            interpolation method for the consecutive 1D interpolations. See
            the scipy.interpolate.interp1d documentation for the possibilities.
            The default is 'linear'.

        Returns
        -------
        None.

        """
        self.xGrid, self.yGrid, self.zGrid = ats.toCanonicalMeshgrid(xGrid, yGrid, zGrid)
        self.kind = kind
        # get minimum amount of points needed to perform interpolation
        # with the selected kind
        minVals = {'quadratic': 3, 'cubic': 4}
        if isinstance(self.kind, int):
            self.nbPts = self.kind+1
        elif self.kind in minVals.keys():
            self.nbPts = minVals[self.kind]
        else:
            self.nbPts = 2

        # pre compute interpolators for all rows in the data
        self.rowInterpolators = [self.get1dInterpolator(x, z) for x, z in zip(self.xGrid, self.zGrid)]


    def __call__(self, xs, ys):
        """
        Perform 2D interpolation at the provided x and y coordinates

        Parameters
        ----------
        xs : float
            x-coordinate at which to interpolate z = f(x,y)
        ys : float
            y-coordinate at which to interpolate z = f(x,y)

        Returns
        -------
        array
            interpolated value corresponding to the provided coordinate
            point

        """
        return self.sample(xs, ys)


    def get1dInterpolator(self, x, y):
        """
        Get 1D interpolator between x and y which may contain NaN values,
        which are simply ignored

        """
        mask = np.logical_not(np.isnan(y))

        kind = self.kind
        if kind == 'cubic' and np.count_nonzero(mask) <= 3:
            kind = 'linear'
        if kind == 'linear' and np.count_nonzero(mask) == 1:
            kind = 'nearest'
        if np.count_nonzero(mask) <= 0:
            return lambda x: np.nan

        return itp.interp1d(x[mask], y[mask], kind=kind,
                            fill_value=np.nan, bounds_error=False)

    def sample(self, xs, ys):
        """
        Perform 2D interpolation at the provided x and y coordinates

        Parameters
        ----------
        xs : float
            x-coordinate at which to interpolate z = f(x,y)
        ys : float
            y-coordinate at which to interpolate z = f(x,y)

        Returns
        -------
        array
            interpolated value corresponding to the provided coordinate
            point

        """
        # get indices of the minimum possible amount of rows around the
        # desired point needed to perform interpolation
        rowInd = np.argmin(np.abs(self.yGrid[:,0]-ys))
        startInd = rowInd - self.nbPts//2
        endInd = rowInd + self.nbPts
        if startInd < 0:
            maskData = lambda x: x[:self.nbPts]
        elif endInd > self.yGrid.shape[0]:
            maskData = lambda x: x[-self.nbPts:]
        else:
            maskData = lambda x: x[startInd:endInd]

        # get intersection between the plane x = xs and the (x,y,z) surface
        xSection = np.array([f(xs) for f in maskData(self.rowInterpolators)])
        # interpolate that intersection for ys
        zFun = self.get1dInterpolator(maskData(self.yGrid[:,0]), xSection)

        return zFun(ys)

    # def interpBilinear(self, x, y):
    #     xVals = self.xGrid[0,:]
    #     yVals = self.yGrid[:,0]

    #     if x <= xVals[0]:
    #         xInd1, xInd2 = (0, 1)
    #     elif x >= xVals[-1]:
    #         xInd1, xInd2 = (len(xVals)-2, len(xVals)-1)
    #     else:
    #         xInd2 = np.where(xVals > x)[0][0]
    #         xInd1 = xInd2 - 1

    #     if y <= yVals[0]:
    #         yInd1, yInd2 = (0, 1)
    #     elif y >= yVals[-1]:
    #         yInd1, yInd2 = (len(yVals)-2, len(yVals)-1)
    #     else:
    #         yInd2 = np.where(yVals > y)[0][0]
    #         yInd1 = yInd2 - 1

    #     # get the grid points most closely surrounding the
    #     # sample point
    #     x1 = xVals[xInd1]
    #     x2 = xVals[xInd2]
    #     y1 = yVals[yInd1]
    #     y2 = yVals[yInd2]
    #     z11 = self.zGrid[yInd1, xInd1]
    #     z21 = self.zGrid[yInd1, xInd2]
    #     z12 = self.zGrid[yInd2, xInd1]
    #     z22 = self.zGrid[yInd2, xInd2]

    #     # perform bilinear interpolation
    #     fy1 = ((x2-x) * z11 + (x-x1) * z21) / (x2-x1)
    #     fy2 = ((x2-x) * z12 + (x-x1) * z22) / (x2-x1)
    #     fxy = ((y2-y) * fy1 + (y-y1) * fy2) / (y2-y1)

    #     return fxy


def liesBetween(test, bound1, bound2, include1=False, include2=False):
    """
    Check whether a value lies between two bounds

    Parameters
    ----------
    test : float
        value to be testes
    bound1 : float
        first bound
    bound2 : float
        second bound
    include1 : boolean, optional
        Marks the first bound of the interval as open (False) or closed (True).
        The default is False.
    include2 : TYPE, optional
        Marks the second boundof the interval as open (False) or closed (True).
        The default is False.

    Returns
    -------
    between : boolean

    """
    between = (bound1 < test < bound2) or (bound1 > test > bound2)
    if include1:
        between = between or test == bound1
    if include2:
        between = between or test == bound2
    return between


def getMonotonicSegmentEnd(x, iStart=0):
    """
    Get the last index belonging to a monotonic sequence of values in x
    
    This index iEnd is such that x[iStart:iEnd+1] increases or decreases
    monotonically so it marks the last entry in a monotonic sequence in
    values of x starting at iStart

    Parameters
    ----------
    x : array
        subsequent values that can be monotonic or non-monotonic
    iStart : integer, optional
        index beyond which to look for the end of the monotonic sequence.
        The default is 0.

    Returns
    -------
    integer
        last index included in the monotonic sequence starting at iStart

    """
    if x[iStart+1] > x[iStart]:
        endInd = np.where(x[iStart+1:] <= x[iStart:-1])[0]
    elif x[iStart+1] < x[iStart]:
        endInd = np.where(x[iStart+1:] >= x[iStart:-1])[0]
    else:
        return getMonotonicSegmentEnd(x, iStart+1)
    if len(endInd) >= 1:
        return iStart + endInd[0]
    else:
        return len(x) - 1


def interp1d(xs, x, y, kind='linear'):
    """
    Interpolate y = f(x) at xs where x can be a nonmonotonic array
    
    In case x is monotonic, perform standard 1d interpolation of all samples
    in xs using the scipy interpolation library with extrapolation enabled.
    In case x is non-monotonic, the function is interpolated at the lowest
    index where it reaches the current xs, but is higher than the index at
    which the previous xs was interpolated. Therefore, the order in which
    the points in xs are defined matters

    Parameters
    ----------
    xs : scalar or 1-dimensional array
        x-values at which to interpolate y = f(x)
    x : 1-dimensional array with n elements
        x-data of the interpolated data set
    y : 1-dimensional array with n elements
        y-data of the interpolated data set
    kind : string, optional
        Interpolation method. See scipy.interpolate.interp1d documentation
        for the accepted options. The default is 'linear'.

    Returns
    -------
    float or np.ndarray
        interpolated results in the same data format as xs

    Examples
    --------
    Necessary imports:

    >>> import interpolationTools as its

    Monotonic interpolation:

    >>> x = [1,2,3,4,5]
    >>> y = [1,4,9,16,25]
    >>> its.interp1d(2.5, x, y)
    6.5

    >>> x = [5,4,2,1]
    >>> y = [1,4,9,5]
    >>> its.interp1d([1.5, 4.5, -.5], x, y)
    array([7. , 2.5, -1.])

    Non-monotonic interpolation

    >>> x = [1,2,3,4,3,2,3,5]
    >>> y = [1,2,3,4,5,6,7,8]
    >>> its.interp1d([2.5, 3.5, 2.5, 2.5, 3.5, 7.], x, y)
    array([2.5 , 3.5 , 5.5 , 6.5 , 7.25, 9.  ])

    """
    # parse input data
    returnAsScalar = not isinstance(xs, (list, tuple, np.ndarray))
    x, y = np.array(x), np.array(y)
    notNan = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    x, y = x[notNan], y[notNan]

    # x is a monotonic array
    if all(x[1:] > x[:-1]) or all (x[1:] < x[:-1]):
        interpolator = itp.interp1d(x, y, kind=kind, fill_value='extrapolate')
        ys = interpolator(xs)

    # x is a non-monotonic array
    else:
        # cast xs into an array
        if returnAsScalar:
            xs = [xs,]
        ys = np.zeros(len(xs))
        # loop over all elements in xs
        segStart = 0
        segEnd = getMonotonicSegmentEnd(x, segStart)
        xPrev = x[segStart]
        for i, sample in enumerate(xs):
            # the next datapoint can not be found within this segment after
            # the previous datapoint, so look for the next segment that
            # includes the data point
            if not liesBetween(sample, xPrev, x[segEnd], include1=i==0):
                while segEnd < len(x)-1:
                    segStart = segEnd
                    segEnd = getMonotonicSegmentEnd(x, segStart)
                    if liesBetween(sample, x[segStart], x[segEnd], True, True):
                        break
            # interpolate the datapoint on the current segment
            ys[i] = interp1d(sample, x[segStart:segEnd+1], y[segStart:segEnd+1], kind)
            xPrev = sample

    # return result as scalar if xs was a scalar
    return float(ys) if returnAsScalar else ys


def interpAll(xs, x, y, kind='linear', nMax=None):
    """
    Interpolate y to find all ys where the nonmonotonic function x(ys) = xs

    Parameters
    ----------
    xs : float
        x value for which to find all matching y values
    x : 1D array
        x-coordinates of datapoints
    y : 1D array
        matching y-coordinates of datapoints
    kind : string, optional
        Interpolation method. See scipy.interpolate.interp1d documentation
        for the accepted options. The default is 'linear'.
    nMax : integer
        Maximum number of results. If this amount of corresponding values is
        found, the interpolator terminates. The default is None which does not
        put an upper bound on the number of solutions.

    Returns
    -------
    ys : 1D np.ndarray
        list of all y-values for which x(ys) becomes approximately equal to xs

    """
    # parse input data
    indexMask = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    x, y = np.array(x)[indexMask], np.array(y)[indexMask]

    # loop over all segments
    ys = []
    segStart = 0
    segEnd = segStart
    while segEnd < len(x)-1:
        segEnd = getMonotonicSegmentEnd(x, segStart)
        # get matching xs if it lies within the segment
        if liesBetween(xs, x[segStart], x[segEnd], include1=True):
            ys.append(interp1d(xs, x[segStart:segEnd+1], y[segStart:segEnd+1], kind))
        segStart = segEnd
        if nMax is not None and len(ys) >= nMax:
            break

    return np.array(ys)


def interpMax(x, y, mode='all', radiusPrev=1, radiusNext=1):
    """
    Find interpolated position and height of local maxima of y = f(x)
    
    The position and height of the maxima are obtained by fitting a
    quadratic polynomial to y = f(x) around datapoints that are locally
    maximum and getting the apex of this quadratic. Different modes can
    be selected to only do this for the first, last or global maximum
    instead of for all local maxima

    Parameters
    ----------
    x : array
        x-coordinates of datapoints
    y : TYPE
        y-coordinates of datapoints
    mode : float, optional
        which maxima to apply the interpolation on. The options are
        - 'all': apply to all local maxima. This is the default.
        - 'global': apply only to the global maximum
        - 'first': apply only to the first (in index) local maximum
        - 'last': apply only to the last (in index) local maximum
    radiusPrev : integer, optional
        number of datapoints to include in the fit before (lower index) the
        local maximum. The default is 1.
    radiusNext : TYPE, optional
        number of datapoints to include in the fit after (higher index) the
        local maximum. The default is 1.

    Returns
    -------
    xMax: float, None or numpy.ndarray
        interpolated x-coordinates of the selected maxima. Array in case the
        mode is 'all'. Otherwise a float, unless no local maxima were found
        as then None is returned
    yMax: float, None or numpy.ndarray
        interpolated y-coordinates of the selected maxima. Same format as
        xMax

    """
    radiusPrev = max(1, int(radiusPrev))
    radiusNext = max(1, int(radiusNext))

    # indices of all data points that are local maxima
    maxInds = np.where(np.logical_and(y[1:-1] > y[:-2], y[1:-1] > y[2:]))[0] + 1

    # select which local maximum to get the position of
    returnScalar = True
    if len(maxInds) == 0:
        return None, None
    elif mode == 'global':
        maxInds = [maxInds[np.argmax(y[maxInds])],]
    elif mode == 'first':
        maxInds = [min(maxInds),]
    elif mode == 'last':
        maxInds = [max(maxInds),]
    else:
        returnScalar = False

    # find location of every selected maximum
    xMaxs = np.zeros(len(maxInds))
    yMaxs = xMaxs.copy()
    for i, maxInd in enumerate(maxInds):
        # fit quadratic function around the first local maximum
        fitInds = range(int(max(0, maxInd-radiusPrev)),
                        int(min(len(x), maxInd+radiusNext+1)))
        params = np.polyfit(x[fitInds], y[fitInds], 2)
        # get coordinates of the peak of the fitted quadratic
        xMaxs[i] = -params[1]/(2*params[0])
        yMaxs[i] = params[2] - params[1]**2/(4*params[0])

    # return locations in desired format
    if returnScalar:
        return xMaxs[0], yMaxs[0]
    else:
        return xMaxs, yMaxs


def interpMin(x, y, mode='all', radiusPrev=1, radiusNext=1):
    """
    Find interpolated position and height of local minima of y = f(x)

    See documentation of interpMax for more information

    """
    xMaxs, yMaxs = interpMax(x, -y, mode, radiusPrev, radiusNext)
    yMaxs = -yMaxs if yMaxs is not None else yMaxs
    return xMaxs, yMaxs