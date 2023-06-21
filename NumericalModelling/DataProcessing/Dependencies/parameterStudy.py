# -*- coding: utf-8 -*-
"""
Class to organize data from an n-dimensional parameter study of which
the results are stored as .json files

Created on Mon Nov 30 11:09:43 2020
Last revision on Thu Sep 9 2021

@author: Bert Van Raemdonck
@email: bert.vanraemdonck@kuleuven.be
@correspondence: benjamin.gorissen@kuleuven.be

requires:
    pickle to cache results
    jsonTools.py (custom) to parse json data

"""

# standard libraries
import os
import pickle
import numpy as np

# custom libraries
from jsonTools import parseJson

class ParameterStudy:

    def __init__(self, path, axisLabels, axisFunctions, overwriteCache=False,
                 verbose=0):
        """
        Object to hold parameter study results in a structured manner

        The resulting python data are cached in the same folder as the
        parameter result source files so they can be retrieved faster on
        running the script again

        Parameters
        ----------
        path : string
            absolute path to the folder where all simulation results of 
            the parameter study are stored
        axisLabels : list of strings
            labels of the parameters that vary throughout the parameter
            study
        axisFunctions : list of functions
            every functon matches an entry in axisLabels. It accepts
            json data in the form of a dictionary and returns the value
            for the coordinate on the corresponding axis
        overwriteCache : boolean, optional
            whether to overwrite any previously cached results (True) or
            to load data from a cache if it exists (False). Data that were
            modified since the last cache are updated automatically. The
            default is False
        verbose : integer, optional
           verbosity level. If <= 0, don't print any status information.
           If > 0, print loading messages for every file. The default is 0.

        """
        self.axisLabels = axisLabels
        self.axisFunctions = axisFunctions
        self.path = path
        self.verbose = verbose

        # unique values of the datapoints along each axis in the same order
        # as in axisLabels
        self.axisValues = [np.array([]) for i in self.axisLabels]
        # number of unique axis values in the same order as in axisLabels
        self.axisValueCount = [np.array([]) for i in self.axisLabels]
        
        # all file names in a plain list
        self.fileNames = []
        # all datapoints in a plain list
        self.flatResults = []
        # all datapoints organized in a grid following canonical meshgrid
        # format, so the axis coordinates of index (i,j,...) in the grid
        # correspond to the values at the same index in the grids generated
        # by np.meshgrid(axisValues[0], axisValues[1], ...)
        self.gridResults = []

        self.cachePath = os.path.join(path, 'parameterStudyCache.pkl')
        if not os.path.exists(self.cachePath) or overwriteCache:
            # read data from json files
            if self.verbose > 0: print('Parameter study reading files...')
            self.loadData()
            # write data to cache
            self.writeCache()
        else:
            # read data from cache
            if self.verbose > 0: print('Parameter study reading cache...')
            self.readCache()

        if self.verbose > 0: print('Parameter study loaded\n')


    def writeCache(self):
        """
        Write the data loaded from the result source files into a cache
        for later quick access
        """
        with open(self.cachePath, 'wb') as cache:
            # only cache the obtained data, not the axislabels etc.
            # this allows the user to submit lambda functions for the
            # axis functions, which are not supported by pickle
            cacheData = {"axisValues": self.axisValues,
                         "axisValueCount": self.axisValueCount,
                         "fileNames": self.fileNames,
                         "flatResults": self.flatResults,
                         "gridResults": self.gridResults}
            pickle.dump(cacheData, cache, pickle.HIGHEST_PROTOCOL)


    def readCache(self):
        """
        Read the data originally coming from the result source files from
        a cache and load any additional data that were modified after the
        that cache was made previously
        """
        with open(self.cachePath, 'rb') as cache:
            try:
                study = pickle.load(cache)
                self.axisValues = study["axisValues"]
                self.axisValueCount = study["axisValueCount"]
                self.fileNames = study["fileNames"]
                self.flatResults = study["flatResults"]
                self.gridResults = study["gridResults"]
            except Exception as e:
                if self.verbose > 0:
                    print('Parameter study unable to read cache, reloading result files instead')
                    print('The error was: ')
                    print(e)
                self.loadData()
                self.writeCache()
                return

        updatedFiles = False
        folderFiles = {f for f in os.listdir(self.path) if f.endswith('.json')}
        # check if any files were modified after the last time the cache
        # was updated
        cacheTime = os.path.getmtime(self.cachePath)
        for file in folderFiles:
            lastUpdated = os.path.getmtime(os.path.join(self.path,file))
            if lastUpdated > cacheTime or file not in self.fileNames:
                if self.verbose > 0: print('Parameter study updating', file)
                if file in self.fileNames:
                    self.removeFile(file)
                self.addFile(file)
                updatedFiles = True
        # check if any files were removed after the last time the cache
        # was updated
        for fileName in self.fileNames:
            if fileName not in folderFiles:
                if self.verbose > 0: print('Parameter study removing', file)
                updatedFiles = True
                self.removeFile(fileName)
        # update the parameter study cache
        if updatedFiles:
            self.restructureData()
            self.writeCache()


    def addFile(self, fileName):
        """
        Add a single file to the parameter study

        Parameters
        ----------
        fileName : float
            name of the file on the parameter study path

        """
        # get absolute path from relative path
        filePath = os.path.join(self.path, fileName)
        print(filePath)
        # store file data
        fileData = parseJson(filePath)
        self.flatResults.append(fileData)
        self.fileNames.append(fileName)
        # get new axis values
        for axisID, axisFn in enumerate(self.axisFunctions):
            axisValue = axisFn(fileData)
            if axisValue not in self.axisValues[axisID]:
                # add this axis value to the list
                self.axisValues[axisID] = np.append(self.axisValues[axisID], axisValue)
                self.axisValueCount[axisID] = np.append(self.axisValueCount[axisID], 1)
            else:
                # increment amount of users of this axis value
                valueInd = np.argmin(np.abs(self.axisValues[axisID]-axisValue))
                self.axisValueCount[axisID][valueInd] += 1


    def removeFile(self, fileName):
        """
        Remove file from the study by file name

        Parameters
        ----------
        fileName : string
            name of the file on to be removed from the parameter study.
            The actual source file on the disk is not deleted

        """
        # remove axis value entry
        for axisID, axisFn in enumerate(self.axisFunctions):
            # get index of the axis value in the axis value list
            axisValue = axisFn(self.flatResults[self.fileNames.index(fileName)])
            valueInd = np.argmin(np.abs(self.axisValues[axisID]-axisValue))
            # decrement the amount of users of this axis value
            self.axisValueCount[axisID][valueInd] -= 1
            # delete axis value if it has no more users
            if self.axisValueCount[axisID][valueInd] <= 0:
                self.axisValues[axisID] = np.delete(self.axisValues[axisID], valueInd)
                self.axisValueCount[axisID] = np.delete(self.axisValueCount[axisID], valueInd)
        # remove data entry
        delInd = self.fileNames.index(fileName)
        del self.fileNames[delInd]
        del self.flatResults[delInd]


    def restructureData(self):
        """
        Sort the values for every axis in ascending order and put all
        datapoints in a grid with coordinates depending on their axis
        values
        """
        # sort all axis labels
        for i in range(len(self.axisValues)):
            sortInds = np.argsort(self.axisValues[i])
            self.axisValues[i] = self.axisValues[i][sortInds]
            self.axisValueCount[i] = self.axisValueCount[i][sortInds]

        # initialize structured results array in a canonical meshgrid array,
        # so the first axis corresponds to the lowest dimension etc.
        self.gridResults = np.full([len(a) for a in self.axisValues][::-1],
                                   None, dtype=object)
        # fill n-dimensional array with indices referring to the data in
        # the flat array list
        for i, data in enumerate(self.flatResults):
            # get coordinates of this datapoint in the n-dimensional grid
            coords = []
            for j in range(len(self.axisFunctions)):
                dataAxisValue = self.axisFunctions[j](data)
                dataAxisInd = np.where(self.axisValues[j] == dataAxisValue)
                coords.append(dataAxisInd)
            # put coordinate of the datapoint in the flat grid at that
            # point in the n-dimensional grid
            self.gridResults[tuple(coords)[::-1]] = data


    def loadData(self):
        """
        Load all results from the parameter study in the object into an
        n-dimensional array where n is the number of provided axes. Each
        dimension of the array corresponds to data of simulations with
        the same parameter for that axis label. The value of that axis
        parameter is stored in the axisValues property
        """
        # collect all data in a simple array and collect values for the
        # axis coordinates
        for file in os.listdir(self.path):
            if file.endswith('.json'):
                self.addFile(file)
        # update gridData
        self.restructureData()


    def getData(self, axisQueryValues={}, outputDimension="flat"):
        """
        Return an n-dimensional array containing all results with the
        provided values for the axis parameters.

        Parameters
        ----------
        axisQueryValues : dictionary, optional
            dictionary with as keys any axis labels and as values the
            values of those axis variables that should be included in the
            returned selection. If an axis is not present in the keys or
            has None for value, all its values are returned. Otherwise,
            an integer or a list of integer can be provided to select
            particular slices of the dataset. The default is {} which
            returns all data points
        outputDimension : string, optional
            dimensionality of the output array. The default is "flat".
            Possibilities are:
            * "flat": fully flattened 1D array
            * "minimal": the dimensionality of the returned array is
                equal to the number of axes that have more than one slice
            * "full": same dimensionality as the full dataset

        Returns
        -------
        selectedData : numpy.ndarray
            requested datapoints

        """
        # build a list with the selected indices for every data axis
        selectedVals = []
        for i, axisLabel in enumerate(self.axisLabels):
            if axisLabel in axisQueryValues.keys() and \
               axisQueryValues[axisLabel] is not None:
                queryValue = axisQueryValues[axisLabel]
                if not hasattr(queryValue, '__iter__'):
                    selectedVals.append([queryValue,])
                else:
                    selectedVals.append(queryValue)
            else:
                selectedVals.append(self.axisValues[i])

        # build an n-dimensional array with the selected data in canonical
        # meshgrid format so the first axis corresponds to the lowest dimension
        # etc.
        selectedData = np.full([len(a) for a in selectedVals][::-1], None, dtype=object)
        # loop over all indices
        currCoords = [0 for i in self.axisLabels]
        while currCoords[0] < len(selectedVals[0]):
            # get coordinates of the datapoint in the full grid
            fullGridCoords = []
            for i in range(len(self.axisLabels)):
                axisValue = selectedVals[i][currCoords[i]]
                fullGridCoords.append(self.getClosestAxisIndex(self.axisLabels[i],axisValue))
            selectedData[tuple(currCoords)[::-1]] = self.gridResults[tuple(fullGridCoords)[::-1]]
            # get next coordinates in the selection grid
            currCoords[-1] += 1
            for i in range(len(currCoords)-1,0,-1):
                # rollover
                if currCoords[i] >= len(selectedVals[i]):
                    currCoords[i-1] += 1
                    currCoords[i] = 0

        if outputDimension == "flat":
            selectedData = np.array([d for d in selectedData.flatten() if d is not None], dtype=object)
            if len(selectedData) == 1:
                selectedData = selectedData[0]
        elif outputDimension == "minimal":
            minimalShape = [s for s in selectedData.shape if s > 1]
            selectedData = selectedData.reshape(minimalShape)
            if len(selectedData) == 1:
                selectedData = selectedData[0]

        return selectedData


    def getAxisPoints(self, axisLabel):
        """
        Get all values present on an axis

        Parameters
        ----------
        axisLabel : string
            label of axis of which to get the set of unique values present
            in the dataset

        Returns
        -------
        axisValues : numpy.ndarray
            1D array with all axis values

        """
        axisInd = self.axisLabels.index(axisLabel)
        return self.axisValues[axisInd]


    def getClosestAxisIndex(self, axisLabel, queryValue):
        """
        Get index of the axis value closest to the provided query

        Parameters
        ----------
        axisLabel : string
            label of axis of which to find the closest index on
        queryValue : float
            value to which a close match must be found

        Returns
        -------
        axisIndex : integer
            index of the axis value closest to the query value

        """
        axisValues = self.getAxisPoints(axisLabel)
        dists = [abs(v-queryValue) for v in axisValues]
        return np.argmin(dists)


    def map(self, mapping, axisQueryValues={}, outputDimension="minimal"):
        """
        Return an n-dimensional array in which each element corresponds
        to the result of a mapping function applied on one of the queried
        datapoints.

        Parameters
        ----------
        mapping : function
            takes a single data point in the parameter as an input and
            maps it to an output
        axisQueryValues : dictionary, optional
            see documentation of ParameterStudy.getData. The default is {}.
        outputDimension : float, optional
            see documentation of ParameterStudy.getData. The default is
            "minimal".

        Returns
        -------
        mappedData : numpy.ndarray
            selected datapoints after mapping the function to it in
            desired dimensionality

        """
        # get data satisfying the axis queries
        selectedData = self.getData(axisQueryValues, outputDimension)
        # wrapper around function to deal with empty data points
        def robustMapping(dataPoint):
            if dataPoint is not None:
                return mapping(dataPoint)
        # apply function to all datapoints
        mappedData = np.vectorize(robustMapping, otypes=[object,])(selectedData)

        try:
            return mappedData.astype(np.float)
        except Exception:
            return mappedData

    def getProp(self, props, axisQueryValues={}, outputDimension="minimal"):
        """
        Return an n-dimensional array in which each element corresponds
        to a property of every one of the queried datapoints.

        Parameters
        ----------
        props : list
            list with either dictionary keys or list indices pointing
            to a property of a datapoint in the parameter study.
            For example, if the user wants to get
                datapoint["prop1"]["prop1.1"]["prop1.1.1"][0],
            for every datapoint, "props" should be
                ["prop1", "prop1.1", "prop1.1.1", 0]
        axisQueryValues : dictionary, optional
            see documentation of ParameterStudy.getData The default is {}.
        outputDimension : dictionary optional
            see documentation of ParameterStudy.getData. The default is
            "minimal".

        Returns
        -------
        numpy.ndarray
            properties of selected datapoints in desired dimensionality

        """
        # custom mapping function
        def mapping(dataPoint):
            selection = dataPoint
            for prop in props:
                if (isinstance(selection, dict) and \
                    prop in selection.keys()) or \
                   (isinstance(selection, (list, tuple, np.ndarray)) and \
                    isinstance(prop, int) and prop < len(selection)):
                    # desired property is available in the data point
                    selection = selection[prop]
                else:
                    # desired property is not available in the data point
                    return np.nan
            return selection

        return self.map(mapping, axisQueryValues, outputDimension)