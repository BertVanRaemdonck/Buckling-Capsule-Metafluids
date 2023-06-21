# -*- coding: utf-8 -*-
"""
Tools to add functionality to matplotlib module and emulate matplotlib
behavior for custom functions

Created on Tue Feb 23 16:02:55 2021
Last revision on Mon July 19 2021

@author: Bert Van Raemdonck
@email: bert.vanraemdonck@kuleuven.be
@correspondence: benjamin.gorissen@kuleuven.be

"""

import sys
import functools
import numpy as np
import scipy.spatial as spt
import matplotlib.pyplot as plt

class DataInspector:
    def __init__(self, xGrid, yGrid, navigatorAxis, selectAction):
        """
        Cursor to select data and display it

        Display a data cursor on an axis featuring gridded data
        which can be navigated by mouseclicks or by keypresses
        of the arrow keys. The grid point to which the cursor is
        currently pointing is passed to a function.

        Parameters
        ----------
        xGrid : numpy.ndgrid
            x-coordinates of the possible cursor locations on
            the axis
        yGrid : numpy.ndgrid
            y-coordinates of the possible cursor locations on
            the axis
        navigatorAxis : matplotlib.pyplot.Axis
            axis on which the cursor will be displayed
        selectAction : function
            function called whenever a new point is selected
            with the cursor. It should accept the selected x-
            and y-coordinate as arguments

        Returns
        -------
        None.

        """
        self.xGrid = xGrid
        self.yGrid = yGrid
        self.navigatorAxis = navigatorAxis
        self.selectAction = selectAction
        # bind functions to figure
        self.navigatorAxis.figure.canvas.mpl_connect('button_press_event', self.selectByClick)
        self.navigatorAxis.figure.canvas.mpl_connect('key_press_event', self.selectByArrows)
        # create cursor
        self.selectedInd = (0,0)
        self.cursor = navigatorAxis.scatter(None, None, 30, 'k', edgecolors='w')
        self.selectDataPoint(self.selectedInd)

    def selectDataPoint(self, ind):
        """
        Select datapoint with provided index

        Move the cursor on the axis and call the select
        action function with the x and y coordinate of the
        selected datapoint. Wrap the provided index in
        case it exceeds the column or row bounds

        Parameters
        ----------
        ind : tuple
            row and column coordinate of the selected
            datapoint

        Returns
        -------
        None.

        """
        # update selected index
        xInd = ind[0] % self.xGrid.shape[0]
        yInd = ind[1] % self.xGrid.shape[1]
        self.selectedInd = (xInd, yInd)
        # update cursor position
        self.cursor.set_offsets([self.xGrid[self.selectedInd],
                                 self.yGrid[self.selectedInd]])
        # perform select action
        self.selectAction(self.xGrid[self.selectedInd],
                          self.yGrid[self.selectedInd])
        # refresh figure to show new cursor location
        self.navigatorAxis.figure.canvas.draw_idle()

    def selectByClick(self, event):
        """
        Select datapoint by mouseclick

        If the mouse is clicked within the axis, select the
        datapoint closest to the clicked location

        Parameters
        ----------
        event : matplotlib.pyplot button_press_event
            button_press_event

        Returns
        -------
        None.

        """
        if event.inaxes == self.navigatorAxis:
            # find datapoint closest (in display coordinates) to the
            # clicked location
            aspectRatio = self.navigatorAxis.get_data_ratio()
            it = np.nditer(self.xGrid, flags=['multi_index'])
            bestInd = (0,0)
            bestDist = np.inf
            for _ in it:
                currDist = (aspectRatio*(event.xdata - self.xGrid[it.multi_index]))**2 + \
                           (event.ydata - self.yGrid[it.multi_index])**2
                if currDist < bestDist:
                    bestDist = currDist
                    bestInd = it.multi_index
            # select that datapoint
            self.selectDataPoint(bestInd)

    def selectByArrows(self, event):
        """
        Select datapoint by using arrow keys

        Parameters
        ----------
        event : matplotlib.pyplot key_press_event
            key_press_event

        Returns
        -------
        None.

        """
        # offsets to apply to the selected index corresponding to the
        # different accepted keycodes. The assumption rows with a
        # higher index appear higher in the figure and columns with a
        # higher index appear more to the right in a figure. This is
        # consistent with e.g. the matplotlib.pyplot.imshow function
        steps = {'up': (1,0), 'down': (-1,0), 'left': (0,-1), 'right': (0,1)}
        if event.key in steps.keys():
            # apply offset to get indices of the selected datapoint
            xInd = self.selectedInd[0] + steps[event.key][0]
            yInd = self.selectedInd[1] + steps[event.key][1]
            # select that datapoint
            self.selectDataPoint((xInd,yInd))
        # clear keypress events
        sys.stdout.flush()


def removeTwin(ax):
    """
    Remove all axes twinned to the provided axis

    Parameters
    ----------
    ax : matplotlib.pyplot.AxisSubplot
        axis of which to remove all twinned axes

    Returns
    -------
    None.

    """
    for twinAx in ax.figure.axes:
        if twinAx.bbox.bounds == ax.bbox.bounds and twinAx is not ax:
            twinAx.remove()

# %% wrappers to suppress or eliminate aspects of standard pyplot functions

def createNewFigure(f=None, axInd=0, axName='ax', sizeInd=None,
                    sizeName='size', sizeDefault=None):
    """
    Call function on a new figure if no axis was specified
    
    Wraps around functions that take a pyplot Axes instance as an
    argument and either passes the argument on in case it is a valid
    Axes and otherwise creates a new one on a new figure and shows it
    after executing the wrapped function

    Parameters
    ----------
    f : function, optional
        function in need of an Axes instance. The default is None
    axInd : integer, optional
        positional index of the Axes argument in the signature of f. The
        default is 0
    axName : string, optional
        key of Axes keyword argument in the signature of f. The default
        is 'ax'
    sizeInd : integer, optional
        positional index of the argument declaring the figure size in
        case a new figure should be created. The default is None, which
        disables this feature in case the function has no size argument
    sizeName : string, optional
        key of the new figure size keyword in the signature of f. The
        default is 'size'
    sizeDefault : tuple, optional
        default value for the new figure size in case none was provided
        in the function arguments. The default is None

    Returns
    -------
    function
        wrapped version of f with the same return value but with the
        automatic creation of a new figure in case there was no Axes
        provided

    """    
    def wrapFunction(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # create new figure and axes if no ax argument was provided
            newFigure = not((axName in kwargs.keys() and isinstance(kwargs[axName], plt.Axes)) or \
                            (len(args) > axInd and isinstance(args[axInd], plt.Axes)))
            if newFigure:
                # get figure size if it was provided
                if sizeInd is not None and sizeName in kwargs.keys() and \
                   isinstance(kwargs[sizeName], tuple):
                    fig = plt.figure(figsize=kwargs[sizeName])
                elif sizeInd is not None and len(args) > sizeInd and \
                     isinstance(args[sizeInd], tuple):
                    fig = plt.figure(figsize=args[sizeInd])
                else:
                    fig = plt.figure(figsize=sizeDefault)
                kwargs[axName] = fig.gca()
            # call the function on the possibly updated axes
            res = function(*args, **kwargs)
            # show the figure if it was newly created
            if newFigure:
                fig.show()
            return res
        return wrapper

    if f is not None:
        return wrapFunction(f)
    else:
        return wrapFunction

def dontShow(funcName):
    """
    Create pyplot function that does not create a new figure

    Returns a function that applies the pyplot function with the
    provided funcName to a figure that is never rendered. This allows
    to get the return argument of that function without spending time
    rendering the accompanying figure

    Parameters
    ----------
    funcName : string
        name of a pyplot Axes method

    Returns
    -------
    function
        function with the same return value as the original pyplot
        function but that renders no figure in the process

    Raises
    ------
    NameError
        in case the provided funcName is not a pyplot Axes method

    """
    def wrapper(*args, **kwargs):
        # put matplotlib in non-interactive mode to suppress figure creation
        wasInteractive = plt.isinteractive()
        plt.ioff()
        # apply function on a temporary, invisible figure
        tmpFig = plt.figure()
        ax = tmpFig.gca()
        result = getattr(ax, funcName)(*args, **kwargs)
        plt.close(tmpFig)
        # put matplotlib back in the original interactive mode
        if wasInteractive:
            plt.ion()
        return result

    if hasattr(plt, funcName):
        return wrapper
    else:
        raise NameError('{} is not a function that can be applied to an Axes instance')


# %% shortcuts for adding features to figures

@createNewFigure(axInd=4, sizeInd=5, sizeName='figsize')
def scatterMarkers(x, y, centerCol='none', centerRatio=.1, ax=None,
                   figsize=None, **kwargs):
    """
    Create scatter plot in which the markers can have up to three colors

    These colors are the marker face color and edge color as usual, plus
    a center color which is rendered on a smaller version of the marker
    with the same center. The color(s) and size(s) of the marker(s) can
    be provided, and apart from that this function works as the regular
    pyplot scatter function

    Parameters
    ----------
    x : list
        x-coordinates of the data points
    y : list
        y-coordinates of the data points
    centerCol : (list of) color specification(s), optional
        color of the patches at the center of the markers. The default is
        'none' which shows no center patch
    centerRatio : (list of) floats, optional
        ratio between the size of the center patch and the size of the
        outer marker. The default is .2.
    ax : matplotlib.subplot.Axes, optional
        axis to create the scatter plot on. The default is None, which
        creates a new figure
    figsize : tuple, optional
        size for the figure if a new one is created. The default is None
    **kwargs : dictionary
        keyword arguments to be passed on to the pyplot scatter function
        without modification

    Returns
    -------
    matplotlib.collections.PathCollection
        same as the pyplot scatter function

    """
    if isinstance(x, np.ndarray):
        x = list(x)
        y = list(y)
    elif not isinstance(x, list):
        x = [x,]
        y = [y,]

    # get default marker settings
    silentScatter = dontShow('scatter')
    defaultScatter = silentScatter(0, 0)
    mainSettings = {
        's': defaultScatter.get_sizes()[0],
        'c': defaultScatter.get_facecolors()[0],
        'linewidths': defaultScatter.get_linewidths()[0],
        'edgecolors': defaultScatter.get_edgecolors()[0]
        }

    # overwrite default settings with provided keyword arguments
    for prop in mainSettings.keys():
        if prop in kwargs.keys():
            mainSettings[prop] = kwargs[prop]
            del kwargs[prop]

    # convert all settings to arrays with the same length as the data
    otherSettings = {'centerCol': centerCol, 'centerRatio': centerRatio}
    for settingDict in (mainSettings, otherSettings):
        for prop, value in settingDict.items():
            if isinstance(value, (list, tuple)):
                if len(value) == len(x):
                    settingDict[prop] = list(value)
                else:
                    settingDict[prop] = [value[0] for i in x]
            else:
                settingDict[prop] = [value for i in x]

    # create new arguments for scatter function
    xDouble = x*2
    yDouble = y*2
    doubleSettings = {}
    doubleSettings['s'] = mainSettings['s'] + [r*s for r,s in zip(otherSettings['centerRatio'], mainSettings['s'])]
    doubleSettings['c'] = mainSettings['c'] + otherSettings['centerCol']
    doubleSettings['linewidths'] = mainSettings['linewidths'] * 2
    doubleSettings['edgecolors'] = mainSettings['edgecolors'] + ['none' for i in x]

    return ax.scatter(xDouble, yDouble, **doubleSettings, **kwargs)

@createNewFigure(axInd=5, sizeInd=6)
def plotReadoutLines(x, y, xLines='bottom', yLines='left', col='k', ax=None,
                     size=None, **kwargs):
    """
    Create lines connecting data points to their coordinate axis values

    Parameters
    ----------
    x : list
        x-coordinates of data points for which to read out the values
    y : list
        y-coordinates of data points for which to read out the values
    xLines : string, optional
        at what side of the plot to draw the readout lines for the x-axis
        * 'bottom' (default)
        * 'top'
        * 'both'
        * 'none'
    yLines : string, optional
        at what side of the plot to draw the readout lines for the y-axis
        * 'left' (default)
        * 'right'
        * 'both'
        * 'none'
    col : matplotlib color specification, optional
        stroke color for the readout lines. The default is 'k'.
    ax : matplotlib.subplot.Axes, optional
        axis to create the scatter plot on. The default is None, which
        creates a new figure
    size : tuple, optional
        size for the figure if a new one is created. The default is None
    **kwargs : dictionary
        keyword arguments to be passed on to the pyplot plot function
        without modification

    """
    if not isinstance(x, (list, tuple, np.ndarray)):
        x = [x,]
        y = [y,]

    for xVal, yVal in zip(x, y):
        # plot readout to the x-axis
        if xLines == 'bottom':
            yPoints = [ax.get_ylim()[0], yVal]
        elif xLines == 'top':
            yPoints = [yVal, ax.get_ylim()[1]]
        elif xLines == 'both':
            yPoints = ax.get_ylim()
        if xLines in ('bottom', 'top', 'both'):
            ax.plot([xVal,xVal], yPoints, col, **kwargs)

        # plot readout to the y-axis
        if yLines == 'left':
            xPoints = [ax.get_xlim()[0], xVal]
        elif yLines == 'right':
            xPoints = [xVal, ax.get_xlim()[1]]
        elif yLines == 'both':
            xPoints = ax.get_xlim()
        if yLines in ('left', 'right', 'both'):
            ax.plot(xPoints, [yVal,yVal], col, **kwargs)

@createNewFigure(axInd=2, sizeInd=3)
def plotConvexHull(x, y, ax=None, size=None, *args, **kwargs):
    notNanCoords = [not(np.isnan(xi) or np.isnan(yi)) for xi, yi in zip(x,y)]
    dataPoints = np.array([[xi, yi] for xi, yi, notnan in zip(x, y, notNanCoords) if notnan])
    # compute convex hull over data points
    hull = spt.ConvexHull(dataPoints)

    # get convex hull edge segments
    facets = hull.simplices
    # sort them in connected order
    sortedFacets = [0,]
    while len(sortedFacets) < len(facets):
        neighbors = hull.neighbors[sortedFacets[-1]]
        sortedFacets.append(neighbors[0] if neighbors[0] not in sortedFacets else neighbors[1])
    sortedFacets = facets[sortedFacets,:].tolist()
    # get list of subsequent vertices defining the convex hull
    sortedVerts = sortedFacets[0] if sortedFacets[0][1] in sortedFacets[1] else sortedFacets[0][::-1]
    for facet in sortedFacets[1:]:
        sortedVerts.append(facet[1] if facet[0] in sortedVerts else facet[0])

    # plot convex hull
    ax.add_patch(plt.Polygon(dataPoints[sortedVerts], *args, **kwargs))