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
# FILL THIS IN:
#	baseFolder is the path on your local file system that serves
#	as the main folder of this script. You can chose any path to
# 	which you have write access. You have to make sure that the
# 	selected baseFolder contains the following three subfolders
# - baseFolder/Scripts
#	this folder contains all the script files that you downloaded,
#	including this very file
# - baseFolder/SimulationFiles
#	this folder is initially empty, but will come to contain all
#	the temporary Abaqus file as it is running which allows you
#	to follow the status of the simulation and debug it.
# - baseFolder/Results
#	this is the folder to which a json file with the simulation
#	results will be written
baseFolder = r'C:\Users\XXXX\Downloads\Buckling-Capsule-Metafluids-main\NumericalModelling'
scriptFolder = os.path.join(baseFolder, r'Scripts')
simFolder = os.path.join(baseFolder, r'SimulationFiles')
outFolder = os.path.join(baseFolder, r'Results')

# import custom functions
# FILL THIS IN:
#	the absolute path to the Dependencies folder should be
#	included here so the python modules can be imported
sys.path.append(scriptFolder)
sys.path.append(r'C:\Users\XXXX\Downloads\Buckling-Capsule-Metafluids-main\NumericalModelling\FEMScripts\Dependencies')

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

# Definition of constants
SOLID = 'solid'
SHELL = 'shell'
AXI = 'axisymmetric'
THREED = '3D'
PIN = 'pin'
AVERAGE = 'average'
WRITEINP = 0
EXECUTE = 1

# FILL THIS IN
#	simControls is the main variable that controls which simulations
#	are run with which variables. It relies on a custom class
#	controlParameters that is defined in the AbaqusScripUtilities.py
#	file in the dependencies folder. It allows to define the
#	parameters for the simulations as a dictionary with as keys the
#	names of the parameters and as values either single values or
#	lists of values. In case a list is provided, one simulation will
#	be run for every value of that parameter in the list. In case a
# 	single value is provided, that parameter will stay constant
#	throughout all simulations. It is possible to enter lists for
# 	multiple parameters, but they should all have the same length.

# 	In all simulations, the midline radius of the capsule is 1 and
#	the shear modulus is 1. The other parameters that have to be
#	defined are the following:
#	- 'eta': shell thickness
#	- 'delta': magnitude of the imperfection. The imperfection is
#		   a dimple shaped as a Gaussian bell curve centered at 
#                  the pole and displacing all points radially inward
#	- 'sigma': standard deviation (in radians) of the bell curve
#                  defining the imperfection
#       - 'ecc': eccentricity of the capsule, namely the distance
#                between the center point of the outer surface of the
#                capsule and the center point of the inner surface of
#                the inner surface. 0 for a concentric shell. Must be
#                below eta
#	- 'alpha': ellipticity parameter, namely the radius between the
#                  minor and the major radius of the capsule in case
#                  the shape is an oblate ellipsoid instead of a sphere.
#                  1 for a perfect sphere
# 	- 'load_max': (approximate) ratio between the maximal change in
#		      volume of the capsule and the initial internal
#   		      volume
# 	- 'element_type': SOLID for solid elements, SHELL for shell
# 			  elements
# 	- 'dimensionality': AXI for an axisymmetric simulation, 3D for
#                           a fully three dimensional simulation
#    	- 'clamping': type of boundary condition to constrain the
#                     rigid-body vertical motion of the capsule. PIN
#                     for pinning the anitpole, AVERAGE for constraining
#                     the average vertical displacement of all nodes
#                     using a distributing coupling constraint
# 	- 'it_max': maximal number of iterations in the simulation
# 	- 'step_max': maximal time increment size in the simulation,
# 		      taking into account that the total step time is 10
# 	- 'mesh_size': maximal seed size for the mesh generation
etas = np.array([2./9.,])
simControls = scrpt.controlParameters({
				'eta': etas,
				'delta': .005*etas,
				'sigma': np.pi/6.,
				'ecc': 0., 
				'alpha': 1.,
				'load_max': [.97 if h < .5 else .75 for h in etas],
				'element_type': [SOLID if eta > .05 else SHELL for eta in etas],
				'dimensionality': AXI,
				'clamping': PIN,
				'it_max': 2500,
				'step_max': .03,
				'mesh_size': [max(eta/20, .03) for eta in etas]
				})

# FILL THIS IN
#	define what to do with the simulations:
#       - EXECUTE executes all simulations one after the other.
#	          the Abaqus interface will not respond until all
#                 simulations have been completed, but the progress
#                 can be tracked through the messages in the Abaqus
#                 terminal or in the .dat file in the SimulationFiles
# 		  folder
# 	- WRITEINP just writes the input file for every simulation
#  	    	   to the SimulationFilesFolder without running the
#		   the simulations.
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
