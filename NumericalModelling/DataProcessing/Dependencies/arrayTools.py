# -*- coding: utf-8 -*-
"""
Functions to process numpy arrays

Created on Fri Jul  9 11:10:52 2021
Last revision on Tue July 9 2021

@author: Bert Van Raemdonck
@email: bert.vanraemdonck@kuleuven.be
@correspondence: benjamin.gorissen@kuleuven.be

"""

import numpy as np


class MeshGridError(Exception):
    def __init__(self, message):
        super().__init__("Not a meshgrid because {}".format(message))

def toCanonicalMeshgrid(xGridIn, yGridIn, *otherGridsIn):
    """
    Check if x and y can form a meshgrid and bring them in canonical form

    In the canonical form, all rows in x are the same and contain
    monotonically increasing values for increasing column indices.
    Similarly, all columns in y are the same and contain monotonically
    increasing values for increasing row indices. x and y can be brought
    into this form through rotation and mirror transforms.
    
    Other grids linked to the x and y grid can be provided to undergo
    the same transformations so the data keeps lining up

    Parameters
    ----------
    xGridIn : mxn numpy.ndarray
        x-coordinate grid
    yGridIn : mxn numpy.ndarray
        y-coordinate grid
    *otherGridsIn : mxn numpy.ndarray, optional
        other data associated with every x,y coordinate pair

    Raises
    ------
    MeshGridError
        when xGridIn and yGridIn can not be simply transformed into a
        canonical meshgrid because their rows/columns are not identical
        or do not contain monotonically changing values

    Returns
    -------
    xGrid : mxn or nxm numpy.ndarray
        same data as xGridIn, but transformed into canonical form
    yGrid : mxn or nxm numpy.ndarray
        same data as yGridIn, but transformed into canonical form
    otherGrids : list of mxn or nxm numpy.ndarrays
        same data as in otherGridsIn, transformed in the same way as
        xGrid and yGrid

    Examples
    --------

    Necessary imports:

    >>> import numpy as np
    >>> import arrayTools as ats

    Input data:

    >>> x = np.array([[2, 2, 2],
                      [1, 1, 1],
                      [0, 0, 0]])
    >>> y = np.array([[3, 4, 5],
                      [3, 4, 5],
                      [3, 4, 5]])
    >>> zGrid = np.array([[2, 5, 8],
                          [1, 4, 7],
                          [0, 3, 6]])
    
    Brind meshgrid into canonical form

    >>> ats.toCanonicalMeshgrid(x, y)
    (array([[0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]]),
     array([[3, 3, 3],
            [4, 4, 4],
            [5, 5, 5]]))

    Transform associated data as well

    >>> ats.toCanonicalMeshgrid(x, y, z)
    (array([[0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]]),
     array([[3, 3, 3],
            [4, 4, 4],
            [5, 5, 5]]),
     array([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]]))


    """
    xGrid = xGridIn.copy()
    yGrid = yGridIn.copy()
    otherGrids = [otherGrid.copy() for otherGrid in otherGridsIn]

    for grid in [yGrid,] + otherGrids:
        if grid.shape != xGrid.shape:
            raise MeshGridError('the different grids have different shapes')

    # check if xGrid and yGrid need to be transposed to form
    # a proper meshgrid (where x has identical rows and
    # y has identical columns)
    sameRows = lambda g: all([all(g[0,:] == g[i,:]) for i in range(g.shape[0])])
    if not sameRows(xGrid) and not sameRows(np.transpose(yGrid)):
        if sameRows(np.transpose(xGrid)) and sameRows(yGrid):
            xGrid = np.transpose(xGrid)
            yGrid = np.transpose(yGrid)
            otherGrids = [np.transpose(otherGrid) for otherGrid in otherGrids]
    if not sameRows(xGrid):
        raise MeshGridError('the rows in xGrid are not identical')
    if not sameRows(np.transpose(yGrid)):
        raise MeshGridError('the columns in yGrid are not identical')

    # put the elements in the grid in ascending order
    if not all(np.diff(xGrid[0,:]) > 0):
        if all(np.diff(xGrid[0,:]) < 0):
            xGrid = np.flip(xGrid, 1)
            otherGrids = [np.flip(otherGrid, 1) for otherGrid in otherGrids]
        else:
            raise MeshGridError('the rows in xGrid do not contain monotonically changing values')
    if not all(np.diff(yGrid[:,0]) > 0):
        if all(np.diff(yGrid[:,0]) < 0):
            yGrid = np.flip(yGrid, 0)
            otherGrids = [np.flip(otherGrid, 1) for otherGrid in otherGrids]
        else:
            raise MeshGridError('the columns in yGrid do not contain monotonically changing values')

    if len(otherGrids) > 1:
        return xGrid, yGrid, otherGrids
    elif len(otherGrids) == 1:
        return xGrid, yGrid, otherGrids[0]
    else:
        return xGrid, yGrid


def padToHigherDim(data, mode='right'):
    """
    Convert nD-array of 1D-lists with varying length into (n+1)D padded array
    
    The slices of the padded array in its first dimension have the same
    shape as the input array but each entry is only a single entry from the
    1D list on the corresponding place in the input array. The size of the
    padded array in its first dimension is equal to the maximum length of
    all 1D lists in the input array.

    Parameters
    ----------
    data : numpy.ndarray
        Data to be padded. Every element in the array can be either a float
        or a 1D list, tuple or numpy array
    mode : string, optional
        Pad direction, can be either 'left' or 'right'. The default is 'right',
        which repeats the last element of the 1D array in the dimensions with
        an index larger than the original array length

    Returns
    -------
    fullData : numpy.ndarray
        Padded data with a dimensionality of one higher than the original
        array and containing only scalars

    Examples
    --------
    Necessary imports:

    >>> import numpy as np
    >>> import arrayTools as ats

    Input data:

    >>> dataIn = np.array([ [1, (2,3)], [np.nan, (4,5,6)] ], dtype=object)
    
    Output:
    
    >>> ats.padToHigherDim(dataIn, mode='right')
    array([ [ [1,   2],
              [nan, 4] ],
            [ [1,   3],
              [nan, 5] ],
            [ [1,   3],
              [nan, 6] ] ], dtype=object)
    
    >>> ats.padToHigherDim(dataIn, mode='left')
    array([ [ [1,   2],
              [nan, 4] ],
            [ [1,   2],
              [nan, 5] ],
            [ [1,   3],
              [nan, 6] ] ], dtype=object)

    """
    # convert all entries in the data array to numpy arrays
    arrayData = np.vectorize(lambda x: np.array(x, ndmin=1), otypes=[np.ndarray,])(data)
    # get maximum size of all those arrays
    newSize = np.max(np.vectorize(lambda x: x.size)(arrayData))
    # pad all arrays to the maximum size
    if mode == 'right':
        padData = lambda x: np.pad(x, (0,newSize-x.size), mode='edge')
    else:
        padData = lambda x: np.pad(x, (newSize-x.size,0), mode='edge')
    paddedData = np.vectorize(padData, otypes=[np.ndarray,])(arrayData)
    # construct higher dimensional array from the padded data
    fullData = np.array([np.vectorize(lambda x: x[i], otypes=[object,])(paddedData) \
                         for i in range(newSize)])
    return fullData

def _isnanObject(x):
    if isinstance(x, type(np.nan)):
        return np.isnan(x)
    else:
        return False
isnanObject = np.vectorize(_isnanObject)

def max(a):
    mask = np.logical_not(isnanObject(a))
    if np.count_nonzero(mask) > 0:
        return np.max(a[mask])
    else:
        return np.nan

def min(a):
    mask = np.logical_not(isnanObject(a))
    if np.count_nonzero(mask) > 0:
        return np.min(a[mask])
    else:
        return np.nan