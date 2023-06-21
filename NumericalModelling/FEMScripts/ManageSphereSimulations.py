# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:05:25 2020

@author: u0123347
"""

# -------------
# Set up script
# -------------

# import python functions
import os
import sys
import numpy as np
import json
import math
import datetime

from abaqusConstants import STANDARD_EXPLICIT

# Set up directory system
baseFolder = r'C:\abaquswork\SphereSimulationMaster'
scriptFolder = os.path.join(baseFolder, r'Scripts')
simFolder = os.path.join(baseFolder, r'SimulationFiles')
outFolder = os.path.join(baseFolder, r'Results\EllipsoidImperfection')

# import custom functions
sys.path.append(scriptFolder)
sys.path.append(r'I:\PhD\Utilities\Python\AbaqusParsers')

import AbaqusScriptUtilities as scrpt
from CreateSphereSimulation import generateSphereModel
from ProcessSphereSimulation import processOdb

# set names for the different files
modName = 'Sphere'
odbLocation = os.path.join(simFolder, '{}.odb'.format(modName))

# ----------------------------
# Define simulation parameters
# ----------------------------

# Everything is de-dimensionalized:
#   eta = thickness/mean radius
#   delta = imperfection dimple magnitude/mean radius
#   sigma = imperfection dimple standard deviation in angle
#   load_max = maximum change in volume/initial volume

SOLID = 'solid'
SHELL = 'shell'
AXI = 'axisymmetric'
THREED = '3D'
PIN = 'pin'
AVERAGE = 'average'

etas = np.arange(.1, 1.5, .1)

# calculate parameters for a shell corresponding to the CT measurement
# all dimensions in µm
posInner = np.array([289.6, 283.6, 269.7])
posOuter = np.array([306.0, 287.7, 269.6])
rInner = 174.85
rOuter = 240.7

rMean = .5*(rInner+rOuter)
t = rOuter-rInner
offset = np.linalg.norm(posInner-posOuter)

eta = t/rMean
ecc = offset/t

#tMin = t - offset
#eta = tMin/rMean

#etas = np.array([.002, .004, .006, .008])
#np.array([1./9.5, 2./9., 2.3/8.85, 2.5/8.75, 3.0/8.5, 4.0/8.0, 4.5/7.75, 5.0/7.5])

etas = np.logspace(-2, 0, 21)[18:19]
etas = np.array([.01*(10**.1)**(2*i) for i in [0,]]) #[0,3,6,9]])[3:]#range(12)])
etas = np.array([0.5011872336272725,])

simControls = scrpt.controlParameters({
				'eta': etas, #0.2265,#[.22, .24, .26, .28, .3],
				'delta': .005*etas, #[3*min(.04, .05*eta) for eta in etas],#.02*eta, #2 * 0.005, #5e-2*etas,
				'sigma': np.pi/6., #[min(radians(60), 1.5/np.sqrt(np.sqrt(1-.5**2)/eta)) for eta in etas], #radians(80),
				'ecc': 0., #0.106, #0.,
				'alpha': 1.,
				'load_max': .999, #[.97 if h < .5 else .72*.75 for h in etas], #.97, #.75,
				'element_type': [SOLID if eta > .05 else SHELL for eta in etas],
				'dimensionality': AXI,
				'clamping': PIN,
				'it_max': 2500,
				'step_max': .03,   # total step time is 10
				'mesh_size': [max(eta/20, .03) for eta in etas] #.5*.06#.055 #[max(.01, min(.06, .125*eta)) for eta in etas]
				})

# constants
WRITEINP = 0
EXECUTE = 1
# simulation controls
task = EXECUTE

# -----------------------
# Define helper functions
# -----------------------

def consoleStatusMessage(messageStr):
	"""Print a message in the (Abaqus) console, consisting out of the provided
	messageStr with formatting to make it stand out as a header"""	
	os.system('echo {}'.format('-'*len(messageStr)))
	os.system('echo {}'.format(messageStr))
	os.system('echo {}'.format('-'*len(messageStr)))
	os.system('echo.')


def outputName(jobName, params):
	"""Create a string containing the jobName, the thickness, the center radius
	and the imperfection magnitude indicating the name of the json file 
	outputed by an analysis. Formatted to be readable by both man and machine"""
	selectedKeys = ['eta', 'delta', 'ecc', 'alpha', 'mesh_size']
	pairs = {k: str(round(params[k],4)).replace('.', 'p') for k in selectedKeys}
	fileName = '{}_{}_{}_'.format(jobName, params['element_type'], params['dimensionality'])
	fileName += '_'.join(['{}_{}'.format(k, v) for k, v in pairs.items()])
	fileName += '.json'
	return fileName


def saveOutput(jsonLocation, data):
	"""Write the provided data to a json file with the provided location, 
	formatted to be consistent and human readable"""
	with open(jsonLocation, 'w') as outfile:
		json.dump(data, outfile, indent=4, sort_keys=True)


# ----------------
# Perform analyses
# ----------------

# loop over all defined simulations
for simID in range(simControls.getNbSimulations()):
	params = simControls.getAll(simID)

	# Initialize output file for this simulation
	sphereData = {}
	sphereData['params'] = {
		'geometry': {
			'eta': params['eta'], 
			'ri': 1 - .5*params['eta'],
			'ro': 1 + .5*params['eta'],
			'V0': 4./3.*np.pi*params['alpha']*(1-.5*params['eta'])**3,
		},
		'imperfection': {
			'delta': params['delta'],
			'sigma': params['sigma'],
			'eccentricity': params['ecc'],
			'aspect': params['alpha']
		},
		'material': {
			'type': 'NeoHooke',
		},
		'simulation': {
			'elements': params['element_type'],
			'step': 'Dynamic',
			'model': params['dimensionality'],
			'control': 'volume',
			'clamping': params['clamping'],
			'starttime': str(datetime.datetime.now()),
			'mesh': params['mesh_size']
		}
	}
	jsonLocation = os.path.join(outFolder, outputName(modName, params))

	# Print status message
	os.chdir(simFolder)
	os.system('echo.')
	consoleStatusMessage('{}/{} - Running {} {} simulation'.format(\
						 simID+1, simControls.getNbSimulations(),
						 params['dimensionality'], params['element_type']))

	# Clear previous models and jobs
	if modName in mdb.models.keys():
		del mdb.models[modName]
	if modName in mdb.jobs.keys():
		del mdb.jobs[modName]
	modeldb = mdb.Model(name=modName, modelType=STANDARD_EXPLICIT)

	# Generate model
	simJob = generateSphereModel(modeldb, params, simID)
	simJob.writeInput(consistencyChecking=OFF)
	# Run simulation
	if task == WRITEINP:
		os.system('echo Input file written')
	elif task == EXECUTE:
		os.system('abaqus job={} ask_delete=OFF interactive'.format(modName)) 
		# Get the simulation results and put them in an ordered dictionary
		sphereData['params']['simulation']['endtime'] = str(datetime.datetime.now())
		odbData = processOdb(odbLocation, sphereData, session)
		for k in odbData.keys():
			sphereData[k] = odbData[k]
		# Write all relevant data
		saveOutput(jsonLocation, sphereData)

	os.system('echo.')