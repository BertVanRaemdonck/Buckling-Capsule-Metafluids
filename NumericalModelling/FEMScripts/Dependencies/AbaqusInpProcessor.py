# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:44:00 2020

@author: Bert Van Raemdonck

Contains functions to parse an Abaqus .inp file into a Python class
"""

import re
import os

class InpPoint:

    def __init__(self, *args):
        if len(args) == 4:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]
            self.label = args[3]
        elif len(args) == 1:
            self.parseString(args[0])


    def __str__(self):
        """Convert point object into text as readable by an Abaqus input
        file"""
        # order of the data elements
        dataElements = [self.label, self.x, self.y, self.z]
        # fixed with occupied by each data element
        dataWidths = [8,16,16,16]
        
        # account for 2D points
        if self.z is None:
            del(dataElements[-1])
        # convert data elements to individual strings
        dataStrings = [str(data) for data in dataElements]
        # concatenate to create one string with padded spaces to get fixed
        # width
        pointStr = ""
        for i in range(len(dataElements)):
            pointStr += " "*max(1,dataWidths[i]-len(dataStrings[i]))
            pointStr += dataStrings[i]
            if i < len(dataElements)-1:
                pointStr += ","        
            
        return pointStr


    def parseString(self, lineString):
        """Interpret a line from an input file describing a point"""
        lineData = lineString.split(",")
        lineData = [data.strip() for data in lineData]
        
        self.label = int(lineData[0])
        self.x = float(lineData[1])
        self.y = float(lineData[2])
        if len(lineData) > 3:
            self.z = float(lineData[3])
        else:
            self.z = None


class InpElement:

    def __init__(self, elType, *args):
        self.type = elType
        if len(args) > 1:
            self.label = args[0]
            self.nodeLabels = args[1:]
        else:
            self.parseString(args[0])


    def getFaceNodes(self, faceLabel):
        """Return a list with in sequence the node labels making up the face
        of the element requested by means of a face label"""
        # keys are regular expressions for element names and values are tuples
        # with subsequent entries being lists of node labels that make up the
        # subsequent faces of the elements
        faceMapping = {
            r'C(\D{2,})3': ((1,2), (2,3), (3,1)),
            r'C(\D{2,})4': ((1,2), (2,3), (3,4), (4,1)),
            r'C(\D{2,})6': ((1,4,2), (2,5,3), (3,6,1)),
            r'C(\D{2,})8': ((1,5,2), (2,6,3), (3,7,4), (4,8,1)),
            r'C3D4': ((1,2,3), (1,4,2), (2,4,3), (3,4,1)),
            r'C3D5': ((1,2,3,4), (1,5,2), (2,5,3), (3,5,4), (4,5,1)),
            r'C3D6': ((1,2,3), (4,6,5), (1,4,5,2), (2,5,6,3), (3,6,4,1)),
            r'C3D8': ((1,2,3,4), (5,8,7,6), (1,5,6,2), (2,6,7,3), (3,7,8,4), (4,8,5,1)),
            r'C3D10': ((1,5,2,6,3,7), (1,8,4,9,2,5), (2,9,4,10,3,6), (3,10,4,8,1,7)),
            r'C3D15': ((1,7,2,8,3,9), (4,12,6,11,5,10), (1,13,4,10,5,14,2,7), (2,14,5,11,6,15,3,8), (3,15,6,12,4,13,1,9)),
            r'C3D20': ((1,9,2,18,6,12,5,17), (5,16,8,15,7,14,6,12), (1,17,5,12,6,18,2,9), (2,18,6,14,7,19,3,10), (3,19,7,15,8,20,4,11), (4,20,8,16,5,17,1,12))}

        # get face mapping of this element
        faceMap = ()
        # check if this element type appears in the keys of faceMapping
        for elType, elMap in faceMapping.items():
            if re.match(elType, self.type) is not None:
                faceMap = elMap
        if len(faceMap) <= 0:
            os.system('echo Face map of element type {} not supported'.format(self.type))
            return None

        # apply facemap to get the nodes corresponding to the desired face
        if isinstance(faceLabel, str):
            # drop letters and convert decimals to int
            faceLabel = int(re.findall(r'\d+', faceLabel)[-1])
        faceNodeIndices = faceMap[faceLabel-1]
        faceNodes = [self.nodeLabels[i-1] for i in faceNodeIndices]

        return faceNodes


    def __str__(self):
        """Convert element object into text as readable by an Abaqus input 
        file"""
        # order of the data elements
        dataElements = [self.label] + self.nodeLabels
        # fixed width in characters occupied by each data element
        dataWidths = [8] + [8]*len(self.nodeLabels)
        
        # convert data elements to individual strings
        dataStrings = [str(data) for data in dataElements]
        # concatenate to create one string with padded spaces to get fixed
        # width
        elementStr = ""
        for i in range(len(dataElements)):
            elementStr += " "*max(1,dataWidths[i]-len(dataStrings[i]))
            elementStr += dataStrings[i]
            if i < len(dataElements)-1:
                elementStr += ","
            
        return elementStr


    def parseString(self, lineString):
        """Interpret a line from an input file describing an element"""
        lineData = lineString.split(",")
        lineData = [data.strip() for data in lineData]
        
        self.label = int(lineData[0])
        self.nodeLabels = [int(data) for data in lineData[1:]]


class InpNodeSet:

    def __init__(self, label, nodes=[]):
        self.label = label
        self.nodes = nodes[:]


    def addNode(self, node):
        """Add Node object to the node set"""
        if node not in self.nodes:
            self.nodes.append(node)


    def getNodeLabels(self):
        """Return a list containing the labels of all nodes in this set"""
        return [n.label for n in self.nodes]


    def __str__(self):
        """Convert node set object into text as readable by an Abaqus input
        file"""
        # number of labels per line (max 16)
        nbPerLine = 16
        # fixed width in characters occupied by each label
        dataWidth = 1 + max([len(str(n.label)) for n in self.nodes])
        setStr = ""

        for i in range(len(self.nodes)):
            nodeLabel = self.nodes[i].label
            setStr += " "*(dataWidth-len(str(nodeLabel))) + str(nodeLabel)
            if (i+1)%nbPerLine == 0:
                setStr += "\n"
            elif i < len(self.nodes)-1:
                setStr += ","

        return setStr


class InpElementSet:

    def __init__(self, label, elements=[]):
        self.label = label
        self.elements = elements[:]


    def addElement(self, element):
        """Add Element object to the element set"""
        if element not in self.elements:
            self.elements.append(element)


    def getElementLabels(self):
        """Return a list containing the labels of all elements in this set"""
        return [e.label for e in self.elements]


    def __str__(self):
        """Convert element set object into text as readable by an Abaqus input
        file"""
        # number of labels per line (max 16)
        nbPerLine = 16
        # fixed width in characters occupied by each label
        dataWidth = 1 + max([len(str(e.label)) for e in self.elements])
        setStr = ""

        for i in range(len(self.elements)):
            elementLabel = self.elements[i].label
            setStr += " "*(dataWidth-len(str(elementLabel))) + str(elementLabel)
            if (i+1)%nbPerLine == 0:
                setStr += "\n"
            elif i < len(self.elements)-1:
                setStr += ","

        return setStr


class InpSurface:

    def __init__(self, label, nodeLabels=[], elements=[], sides=[]):
        """Define a surface with a provided label. For a node based surface
        definition, only a list of integer node labels has to be provided. For
        an element based surface definition, either a list of integer element 
        labels or a list of Element objects can be provided. In the latter
        case, a nodal surface definition will be included automatically. In
        either case, a list of strings defining the element sides has to be 
        provided."""
        self.label = label
        self.nodeLabels = []
        self.elementLabels = []
        self.sides = []

        self.addFaces(nodeLabels, elements, sides)


    def addFaces(self, nodeLabels=[], elements=[], sides=[]):
        """Add faces to the surface. For a node based surface definition, only
        a list of integer node labels has to be provided. For an element based
        surface definition, either a list of integer element labels or a list
        of Element objects can be provided. In either case, a list of strings
        defining the element sides has to be provided"""
        if len(elements) > 0:
            # element based surface definition
            if len(sides) == len(elements):
                # make sure sides are formatted as integers
                for side in sides:
                    if isinstance(side, str):
                        self.sides.append(int(re.findall(r'\d+',side)[-1]))
                    elif isinstance(side, int):
                        self.sides.append(side)
                # get elements
                if isinstance(elements[0], InpElement):
                    # generate list of surface nodes
                    for i in range(len(elements)):
                        self.elementLabels.append(elements[i].label)
                        faceNodes = elements[i].getFaceNodes(sides[i])
                        for faceNode in faceNodes:
                            if faceNode not in self.nodeLabels:
                                self.nodeLabels.append(faceNode)
                else:
                    self.elementLabels.extend(elements)
            else:
                os.system("echo Number of side labels must match number of elements")
        elif len(nodeLabels) > 0:
            # node based surface definition
            self.nodeLabels.extend(nodeLabels)


    def __str__(self, mode="NODE"):
        """Convert surface object into text as readable by an Abaqus input
        file. Optional mode argument allows to toggle between a node based
        and an element based surface definition string if available"""
        
        surfaceStr = ""
        if mode.lower() != "node" and len(self.elementLabels) > 0:
            # element surface definition
            for i in range(len(self.elementLabels)):
                surfaceStr += "{}, S{}".format(self.elementLabels[i], self.sides[i])
        else:
            # nodal surface definition
            for nodeLabel in self.nodeLabels:
                surfaceStr += "{}, 1.0\n".format(nodeLabel)

        return surfaceStr


class InpFile:

    def __init__(self, fileName):
        self.name = fileName
        self.file = open(fileName, "r")
        self.lines = self.file.readlines()
        self.file.close()

        self.nodes = []
        self.refPoints = []
        self.elements = []
        self.getGeometry()

        self.nsets = []
        self.elsets = []
        self.getSets()

        self.surfaces = []
        self.getSurfaces()


    def getGeometry(self):
        """Extract the geometric data from the input file. Update the lists
        with node, elements and reference points"""
        lineIsNode = False
        lineIsElement = False
        elementType = None

        points = []

        # process all lines in the file
        for line in self.lines:
            # update data
            if lineIsNode and "*" not in line:
                points.append(InpPoint(line))   
            if lineIsElement and "*" not in line:
                self.elements.append(InpElement(elementType, line))

            # update metadata
            if len(line) >= 2 and \
               line.strip()[0] == "*" and line.strip()[1] != "*":
                # this line is a command line
                lineIsNode = len(line.strip())>=5 and \
                             line.strip().lower() == "*node" 
                lineIsElement = len(line.strip())>=9 and \
                                line.strip().lower()[:9] == "*element,"
                # get element type
                if lineIsElement:
                    lineData = line.split(",")
                    for data in lineData:
                        if "type=" in data:
                            elementType = data.split("=")[-1].strip()

        # separate points into part nodes and reference points
        nodesInEls = set([])
        for element in self.elements:
            for nodeID in element.nodeLabels:
                nodesInEls.add(nodeID)
        for point in points:
            if point.label in nodesInEls:
                self.nodes.append(point)
            else:
                self.refPoints.append(point)


    def getSets(self):
        """Extract the data on node and elements sets from the input file
        and update the the lists of the object"""
        lineIsNset = False
        lineIsElset = False
        lineIsGenerator = False

        # process all lines in the file
        for line in self.lines:
            lineData = [data.strip() for data in line.split(",")]

            # update metadata
            if len(lineData[0]) >= 2 and \
               lineData[0][0] == "*" and lineData[0][1] != "*":
                # this line is a command line
                # update flags
                lineIsNset = "*nset" in lineData[0].lower()
                lineIsElset = "*elset" in lineData[0].lower()
                lineIsGenerator = "generate" in [data.lower() for data in lineData]
                # get set name
                if lineIsNset or lineIsElset:
                    setLabel = [arg.split("=")[1] for arg in lineData if "set=" in arg][0]
                # create new set if it doesn't exist yet
                if lineIsNset and setLabel not in [ns.label for ns in self.nsets]:
                    self.nsets.append(InpNodeSet(setLabel))
                if lineIsElset and setLabel not in [es.label for es in self.elsets]:
                    self.elsets.append(InpElementSet(setLabel))

            # update data
            elif "*" not in lineData[0]:
                # this line is a data line
                if lineIsGenerator:
                    # write out generator line in full
                    generatedLabels = list(range(int(lineData[0]),int(lineData[1])+1,int(lineData[2])))
                    lineData = [str(label) for label in generatedLabels]
                if lineIsNset:
                    # add all nodes on this line to the current node set
                    for label in lineData:
                        if label.isdigit():
                            # this label refers to a single node
                            self.getNodeSet(setLabel).addNode(self.getNode(label))
                        elif len(label) > 0:
                            # this label refers to a previously defined node
                            # set. Add all nodes present in that set
                            for node in self.getNodeSet(label).nodes:
                                self.getNodeSet(setLabel).addNode(node)
                if lineIsElset:
                    # add all elements on this line to the current element set
                    for label in lineData:
                        if label.isdigit():
                            # this label refers to a single element
                            self.getElementSet(setLabel).addElement(self.getElement(label))
                        elif len(label) > 0:
                            # this label refers to a previously defined element
                            # set. Add all elements present in that set
                            for element in self.getElementSet(label).elements:
                                self.getElementSet(setLabel).addElement(element)


    def getSurfaces(self):
        """Extract the data on surfaces from the input file and update the the 
        lists of the object"""
        lineIsSurface = False
        isNodeBased = False

        # process all lines in the file
        for line in self.lines:
            lineData = [data.strip() for data in line.split(",")]

            # update metadata
            if len(lineData[0]) >= 2 and \
               lineData[0][0] == "*" and lineData[0][1] != "*":
                # this line is a command line
                # update flags
                lineIsSurface = "*surface" in lineData[0].lower()
                # get surface name
                if lineIsSurface:
                    nameArgument = [arg for arg in lineData if len(re.findall(r'name(\s*)=',arg.lower()))>0]
                    if len(nameArgument) > 0:
                        surfaceLabel = re.split(r'name(\s*)=',nameArgument[-1].lower())[-1].strip()
                # check whether surface definition is node or element based
                typeArgument = [arg for arg in lineData if len(re.findall(r'type(\s*)=',arg.lower()))>0]
                if len(typeArgument) > 0 and 'node' in typeArgument[0].lower():
                    isNodeBased = True
                # create new surface if it doesn't exist yet
                if lineIsSurface and surfaceLabel not in [s.label for s in self.surfaces]:
                    self.surfaces.append(InpSurface(surfaceLabel))

            # update data
            elif "*" not in lineData[0] and lineIsSurface:
                # this line is a data line
                if isNodeBased:
                    # node based surface definition
                    nodeLabels = []
                    if not lineData[0].isdigit():
                        # extract node labels from provided node set
                        nodeSet = self.getNodeSet(lineData[0])
                        if nodeSet is not None:
                            nodeLabels = nodeSet.getNodeLabels()
                    else:
                        nodeLabels = [int(lineData[0])]
                    # add nodes to surface
                    self.getSurface(surfaceLabel).addFaces(nodeLabels=nodeLabels)
                else:
                    # element based surface definition
                    elementLabels = []
                    elementFaces = []
                    if not lineData[0].isdigit():
                        # extract element labels from provided element set
                        elementSet = self.getElementSet(lineData[0])
                        if elementSet is not None:
                            elementLabels = elementSet.getElementLabels()
                            elementFaces = [lineData[1]]*len(elementLabels)
                    else:
                        elementLabels = [int(lineData[0])]
                        elementFaces = [lineData[1]]
                    # add elements to surface
                    surfaceElements = [self.getElement(el) for el in elementLabels]
                    self.getSurface(surfaceLabel).addFaces(elements=surfaceElements, sides=elementFaces)


    def getObjectByLabel(self, objectCollection, label):
        obj = [o for o in objectCollection if o.label == label]
        if len(obj) > 1:
            os.system("echo Multiple objects with label {} found".format(label))
        elif len(obj) < 1:
            os.system("echo No object with label {} found".format(label))
        else:
            return obj[0]


    def getNode(self, nodeLabel):
        """Return the Node object corresponding to the given label"""
        return self.getObjectByLabel(self.nodes + self.refPoints, int(nodeLabel))


    def getElement(self, elementLabel):
        """Return the Element object corresponding to the given label"""
        return self.getObjectByLabel(self.elements, int(elementLabel))


    def getNodeSet(self, setLabel):
        """Return the NodeSet object corresponding to the given label"""
        return self.getObjectByLabel(self.nsets, setLabel)


    def getElementSet(self, setLabel):
        """Return the ElementSet object corresponding to the given label"""
        return self.getObjectByLabel(self.elsets, setLabel)


    def getSurface(self, surfaceLabel):
        """Return the Surface object corresponding to the given label"""
        return self.getObjectByLabel(self.surfaces, surfaceLabel)


    def getMaxPointID(self):
        """Return the heighest ID occupied by either a node or a reference
        point"""
        maxNodeID = max([n.label for n in self.nodes])
        maxRefPointID = max([r.label for r in self.refPoints])

        return max([maxNodeID, maxRefPointID])


    def getMaxElementLabel(self):
        """Return the highest element label"""
        return max([e.label for e in self.elements])