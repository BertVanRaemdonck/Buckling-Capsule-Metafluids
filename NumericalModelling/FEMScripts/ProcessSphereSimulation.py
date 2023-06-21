# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 09:53:52 2021

@author: Bert Van Raemdonck
@e-mail: bert.vanraemdonck@kuleuven.be

Process the results of a single sphere simulation

"""

import datetime
import numpy as np
import AbaqusOdbProcessor as out

from odbAccess import *
from abaqusConstants import *

def processOdb(odbLocation, sphereData, session):
	"""
	Process the result of the dynamic simulation with odb on the provided
	odbLocation. Returns a structure containing the timestamp of the
	simulation and the pressure, volume and pole displacement, separated in 
	inflation and deflation
	"""
	resultData = {"timestamp": str(datetime.datetime.now())}

	# open the results of the simulation
	results = out.OdbFile(odbLocation)

	# get internal name of the shell
	odb = session.openOdb(odbLocation)
	shellName = [i for i in odb.rootAssembly.instances.keys() if 'shell' in i.lower()][0]
	odb.close()

	# get history data
	p = results.getHistoryData('PCAV', regionName='Assembly.ShellCavityRP'.upper(), matchExact=True)
	v = results.getHistoryData('CVOL', regionName='Assembly.ShellCavityRP'.upper(), matchExact=True)
	A = results.getHistoryData('CAREA')
	Ax = results.getHistoryData('XN1')
	Ay = results.getHistoryData('XN2')
	ut = results.getHistoryData('U2', regionName='{}.Pole'.format(shellName).upper(), matchExact=True)
	ub = results.getHistoryData('U2', regionName='{}.Antipole'.format(shellName).upper(), matchExact=True)

	if sphereData['params']['simulation']['elements'] == 'shell' and \
	   sphereData['params']['simulation']['model'] == 'axisymmetric':
		vData = list(getShellVolume(odbLocation, session))
	else:
		vData = v.data

	# process and put data in dictionary
	vFac = 2. if sphereData["params"]["simulation"]["model"] == "3D" else 1.
	vOffset = sphereData["params"]["geometry"]["V0"] - vFac*vData[0]
	dataEntries = {
		'p': [-pEl for pEl in p.data],
		'v': [vFac*vEl + vOffset for vEl in vData],
		'd': [ubEl - utEl for ubEl, utEl in zip(ub.data, ut.data)],
		'A': [vFac*aEl for aEl in A.data],
		'Ax': [Axel for Axel in Ax.data],
		'Ay': [Ayel for Ayel in Ay.data],
		't': p.independent
	}

	# find index at which inflation ends
	endInfl = np.argmin(v.data)

	# split all data up in inflation and deflation stroke
	# with the turning point included in both strokes
	phases = {
		'infl': [0,endInfl+1],
		'defl': [endInfl,len(dataEntries['p'])]
	}
	for dataName, dataValue in dataEntries.items():
		resultData[dataName] = dict([])
		for phase, bounds in phases.items():
			resultData[dataName][phase] = dataValue[bounds[0]:bounds[1]]

	return resultData


def getShellVolume(odbLocation, session):
	odb = openOdb(odbLocation)
	session.viewports.values()[0].setValues(displayedObject=odb)

	allData = []
	# save all field output data as history output data because otherwise
	# the shell thickness is not processed correctly...
	for var in [(('COORD', NODAL, ((COMPONENT, 'COOR1'),)),),
				(('COORD', NODAL, ((COMPONENT, 'COOR2'),)),),
				(('STH', INTEGRATION_POINT),)]:
		allData.append(session.xyDataListFromField(odb=odb, variable=var,
												   outputPosition=NODAL,
												   nodeSets=("SHELL-1.SECTION", )))
	xData = allData[0]
	yData = allData[1]
	tData = allData[2]
	nbNodes = len(xData)
	nbFrames = len(xData[0].data)

	cavityVol = np.zeros(nbFrames)
	xInner = np.zeros(nbNodes)
	yInner = np.zeros(nbNodes)
	for j in range(nbFrames):
		# get coordinates of all nodes on the inner surface
		for i in range(nbNodes):
			# calculate angle of the normal vector
			if i > 0 and i < nbNodes-1:
				xPrev = xData[i-1].data[j][1]
				yPrev = yData[i-1].data[j][1]
				xNext = xData[i+1].data[j][1]
				yNext = yData[i+1].data[j][1]
				angle = np.arctan2(yNext-yPrev, xNext-xPrev) + .5*np.pi
			elif i == 0:
				angle = .5*np.pi
			else:
				angle = -.5*np.pi
			# calculate position of the point on the inner surface
			xMid = xData[i].data[j][1]
			yMid = yData[i].data[j][1]
			t = tData[i].data[j][1]
			xInner[i] = xMid + .5*t*np.cos(angle)
			yInner[i] = yMid + .5*t*np.sin(angle)
		# calculate total volume enclosed
		v = 0
		for i in range(nbNodes-1):
			v += np.pi*(.5*(xInner[i]+xInner[i+1]))**2*(yInner[i+1]-yInner[i])
		cavityVol[j] = v

	odb.close()
	print(cavityVol)

	return cavityVol

