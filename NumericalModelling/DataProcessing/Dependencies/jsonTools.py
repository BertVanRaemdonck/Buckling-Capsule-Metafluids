# -*- coding: utf-8 -*-
import json
import numpy as np

def parseJson(path, data=None):
    """
    Read json file and save every list and tuple as an numpy array for easy
    processing. This happens recursively through the data function argument
    """
    if path is not None:
        jsonFile = open(path)
        jsonLines = jsonFile.read()
        jsonFile.close()
        jsonData = json.loads(jsonLines)
        jsonData = parseJson(None, jsonData)
    else:
        if type(data) is dict:
            jsonData = {k:parseJson(None,v) for k,v in data.items()}
        elif type(data) is list or type(data) is tuple:
            jsonData = np.array(data)
        else:
            jsonData = data
            
    return jsonData

def parseRPT(path, useNumpy=True):
    """
    Read Abaqus report file and save every column as a dictionary entry with
    as key the variable name and as value a numpy array with the data
    """
    offset = 8
    width = 19
    
    rptData = {}
    headers = []
    headersDone = False
    nbCols = 0
    with open(path, 'r') as rpt:
        for i, line in enumerate(rpt):
            if not headersDone:
                if not headers and line.strip():
                    # set up headers
                    nbCols = int((len(line)-offset)/float(width))
                    headers = ["" for j in range(nbCols)]
                if headers and line.strip():
                    # expand headers when they are on multiple lines
                    for col in range(nbCols):
                        headers[col] += line[col*width+offset:(col+1)*width+offset].strip()
                elif headers and not line.strip():
                    # headers are complete
                    headersDone = True
                    rptData = {h:[] for h in headers}
            else:
                # add numerical data
                for h, e in zip(headers, line.split()):
                    rptData[h].append(float(e))

    # convert all to numpy arrays
    if useNumpy:
        for header in rptData.keys():
            rptData[header] = np.array(rptData[header])

    return rptData

def rptToJson(rptLocation, parameters=None):
    """
    Create a json file on the same location and with the same name as the
    provided rpt file. Additional parameters can be provided to include in the
    json file
    """
    rptLocation = rptLocation.strip()
    if rptLocation[-4:] == '.rpt':
        # parse rpt file
        rpt = parseRPT(rptLocation, useNumpy=False)
        # add additional parameters
        if parameters is not None:
            rpt['params'] = parameters
        # write to json file in the same location
        jsonLocation = rptLocation[:-4] + ".json"
        with open(jsonLocation, 'w') as jsonFile:
            json.dump(rpt, jsonFile, indent=4, sort_keys=True)
    else:
        print('RPT to JSON conversion error: \n' + \
              '\tOnly paths to .rpt files are accepted as inputs.\n' + \
              '\tInstead, "{}" was provided'.format(rptLocation))



class TDF:

    def __init__(self, tdfPath, header=True, delimiter='\t'):
        self.path = tdfPath
        self.header = header
        self.delimiter = delimiter
        self.data = self.parseData(self.header)

    def getLines(self):
        """Read the lines of the tdf"""
        with open(self.path) as tdfFile:
            lines = tdfFile.readlines()

        return lines

    def parseData(self, header=None):
        """Parse tdf into a dictionary (if a header exists) or into an array
        (if no header exists or if False is provided)"""
        if header is None:
            header = self.header

        lines = self.getLines()
        
        dataArray = None #np.array([])
        dataNames = []

        # put data in array
        i = 0
        for line in filter(lambda line: len(line.strip()) > 0, lines):
            lineData = line.split(self.delimiter)
            # parse header
            if header:
                dataNames = [d.strip('\n') for d in lineData]
                header = False
            # parse data
            else:
                dataRow = []
                dtype = float
                for d in lineData:
                    if len(d.strip('\n')) > 0:
                        try:
                            dataRow.append(float(d))
                        except Exception:
                            dataRow.append(d)
                            dtype = object
                    else:
                        dataRow.append(np.nan)
                #dataRow = [float(d) if len(d.strip('\n')) > 0 else np.nan for d in lineData]
                if dataArray is None:
                    dataArray = np.empty((len(lines), len(dataRow)), dtype=dtype)
                dataArray[i,:] = dataRow
                i += 1

        # transform array into dictionary
        if len(dataNames) > 0:
            tdfData = {name: dataArray[:i,j] for j, name in enumerate(dataNames)}
        else:
            tdfData = dataArray

        return tdfData

    def __getitem__(self, key):
        return self.data[key]
