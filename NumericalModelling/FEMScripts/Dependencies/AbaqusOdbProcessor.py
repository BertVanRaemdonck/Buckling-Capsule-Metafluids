# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:36:33 2020

@author: u0123347
"""

# this file should be called from within the abaqus software!
# so either through "run script" in the abaqus GUI
# or in the command line through "abaqus cae noGUI=AnaqusOdbProcessor.py"
# or by another script called in either of those two ways

from odbAccess import *
import re

class HistoryOutp:

    def __init__(self, name, setName, step, data):
        self.name = name
        self.region = setName
        self.step = step
        self.independent = [d[0] for d in data]
        self.data = [d[1] for d in data]


class LinearPertb:

    def __init__(self, name, step, data):
        self.name = name
        self.step = step
        self.data = data


class FieldOutp:

    def __init__(self, name, step, frames=dict([])):
        self.name = name
        self.step = step
        self.frames = frames


    def addFrame(self, frameLabel, frameData):
        self.frames[str(frameLabel)] = frameData


    def getFrame(self, frameLabel):
        return self.frames[str(frameLabel)]


class OdbFile:

    def __init__(self, fileName, getFieldOutput=False):
        self.name = fileName
        self.file = openOdb(fileName)
        self.success = 'error' not in self.file.diagnosticData.jobStatus.name.lower()

        self.historyOutputs = self.getHistoryOutputs()
        self.historyOutputs += self.getNegativeEigenvalues()
        self.linearPertbs = self.getLinearPerturbation()
        if getFieldOutput:
            # this can make things very slow because the large volume of data!
            self.fieldOutputs = self.getFieldOutputs() 

        self.file.close()


    def getHistoryOutputs(self):
        historyOutputList = []
        for stepName,step in self.file.steps.items():
            for regionName,region in step.historyRegions.items():
                for dataName,data in region.historyOutputs.items():
                    if data.data is not None:
                        # find the original name of the set this data is
                        # associated to
                        setName = self.getSetsContainingRegion(regionName)
                        historyOutputList.append(HistoryOutp(dataName, setName, stepName, data.data))

        return historyOutputList

    def getSetsContainingRegion(self, regionName):
        """
        Names of history regions in an odb file contain no
        reference to the original name of the matching node
        or element set. Return a list of all possible
        original node/element set names that match with the
        provided history region name

        """
        # parse region name for info on the parent instance etc.
        match = re.search(r'(\w+)\s(.+)\.(\d+)', regionName)
        if match is not None:
            regionType = match.group(1)
            regionInst = match.group(2)
            regionID = match.group(3)
        else:
            print('Could not parse region name {}'.format(regionName))
            return [""]

        # get the parent instance object
        if regionInst.lower() == 'ASSEMBLY'.lower():
            inst = self.file.rootAssembly
        else:
            inst = [v for k,v in self.file.rootAssembly.instances.items() if k.lower() == regionInst.lower()]
            if len(inst) > 0:
                inst = inst[0]
            else:
                print('No instance found matching the region name {}'.format(regionName))
                return [""]

        # find set that contains the right region ID
        setNames = [""]
        # in case the required region is a node
        if 'Node'.lower() in regionType.lower():
            for setName, nodeSet in inst.nodeSets.items():
                nodes = nodeSet.nodes
                if regionInst.lower() == 'ASSEMBLY'.lower():
                    nodes = nodes[0]
                for node in nodes:
                    if int(node.label) == int(regionID) and setName not in setNames:
                        setNames.append('{}.{}'.format(inst.name, setName))
        # in case the required region is an element
        if 'Element'.lower() in regionType.lower():
            for setName, elementSet in inst.elementSets.items():
                elements = nodeSet.elements
                if regionInst.lower() == 'ASSEMBLY'.lower():
                    elements = elements[0]
                for element in elements:
                    if int(element.label) == int(regionID) and setName not in setNames:
                        setNames.append('{}.{}'.format(inst.name, setName))

        return setNames



    def getNegativeEigenvalues(self):
        eigData = []
        diagnData = self.file.diagnosticData
        
        for step in self.file.steps.values():
            if step.number-1 < len(diagnData.steps):
                stepData = diagnData.getStep(step.number-1)
                stepEigs = [(0,None)]
                for inc in range(stepData.numberOfIncrements):
                    incData = stepData.getIncrement(inc)
                    attemptData = incData.attempts[-1]
                    itData = attemptData.iterations[-1]
                    problems = itData.numericalProblemSummary
                    stepEigs.append((inc+1, problems.numberOfNegativeEigenvalues))
                if len(stepEigs) > 1:
                    eigData.append(HistoryOutp('NEGEIG', 'Model', step.name, stepEigs))
        
        return eigData


    def getLinearPerturbation(self):
        linearPertbList = []
        for stepName, step in self.file.steps.items():
            dataList = []
            if len(step.historyRegions) == 0:
                for frame in [step.frames[i] for i in range(1,len(step.frames))]:
                    dataStr = frame.description
                    dataStr = dataStr.split(":")[-1]
                    name, value = [data.strip() for data in dataStr.split("=")]
                    dataList.append(float(value))
                if len(dataList) > 0:
                    linearPertbList.append(LinearPertb(name.strip(), stepName, dataList))

        return linearPertbList


    def getFieldOutputs(self):
        fieldOutputList = []
        for stepName,step in self.file.steps.items():
            fieldNames = step.getFrame(0).fieldOutputs.keys()
            for fieldName in fieldNames:
                fieldOutput = FieldOutp(fieldName.strip(), stepName)
                for frameID in range(len(step.frames)):
                    frameData = dict([])
                    fieldFrame = step.getFrame(frameID).fieldOutputs[fieldName]
                    for fieldFrameVal in fieldFrame.values:
                        if fieldFrameVal.position == "NODAL":
                            label = fieldFrameVal.nodeLabel
                        else:
                            label = fieldFrameVal.elementLabel
                        frameData[str(label)] = fieldFrameVal.data
                    fieldOutput.addFrame(frameID, frameData)
                fieldOutputList.append(fieldOutput)

        return fieldOutputList


    def getHistoryData(self, dataName, stepName=None, instanceName=None, regionName=None, matchExact=False):
        if matchExact:
            equals = lambda a, b: a == b
        else:
            equals = lambda a, b: a.lower() in b.lower()

        queriedData = [h for h in self.historyOutputs if equals(dataName, h.name)]

        if stepName is not None:
            queriedData = [h for h in queriedData if equals(stepName, h.step)]
        if instanceName is not None:
            queriedData = [h for h in queriedData if 
                           any([equals(instanceName, r) for r in h.region])]
        if regionName is not None:
            queriedData = [h for h in queriedData if \
                           any([equals(regionName, r) for r in h.region])]

        if len(queriedData) <= 0:
            print("no data named {} found in step {} in region {}".format(dataName, stepName, regionName))
        elif len(queriedData) > 1:
            print("multiple data found. Specify step name and/or region name")
        else:
            return queriedData[0]


    def getPerturbationData(self, index):
        if len(self.linearPertbs) > 0:
            return self.linearPertbs[0].data[index]
        else:
            return None


    def getFieldDataValue(self, dataName, increment, label, stepName=None):
        """Get the value of the field data with the provided name at the 
        provided increment at the provided node or element label. If the field
        output is requested for multiple steps, the desired step name should be
        provided as well. To request multiple labels at once, a list can be
        provided and the results will be returned in a list with matching
        order"""
        # open the odb again because storing all data takes too long so better
        # to retrieve data on a request basis only
        self.file = openOdb(self.name)

        # if no step name is provided, find the step in which the requested
        # data appears
        if stepName is None:
            stepName = [s.name for s in self.file.steps.values() if dataName in s.getFrame(0).fieldOutputs.keys()]
            if len(stepName) <= 0:
                print("no data found satisfying query")
                self.file.close()
                return None
            elif len(stepName) > 1:
                print("multiple data found. Specify step name")
                self.file.close()
                return None
            else:
                stepName = stepName[0]

        # get field data object
        frameData = self.file.steps[stepName].frames[increment]
        fieldData = frameData.fieldOutputs[dataName].values
        fieldDataValues = []

        # get values of the field data for all requested labels
        if isinstance(label, int) or isinstance(label, str):
            labels = [label,]
        else:
            labels = label
        for l in labels:
            if str(fieldData[0].position) == "NODAL":
                fieldDataValues.append([f.data for f in fieldData if int(f.nodeLabel) == int(l)][0])
            else:
                fieldDataValues.append([f.data for f in fieldData if int(f.elementLabel) == int(l)][0])
        if isinstance(label, int) or isinstance(label, str):
            fieldDataValues = fieldDataValues[0]

        self.file.close()

        return fieldDataValues
