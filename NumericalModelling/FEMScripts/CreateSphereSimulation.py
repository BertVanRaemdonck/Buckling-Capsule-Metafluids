# -*- coding: utf-8 -*-
"""
Created on Thu April 29 09:27:39 2021

@author: Bert Van Raemdonck
@e-mail: bert.vanraemdonck@kuleuven.be

Set up a dynamic simulation of a hollow spherical shell
subject to volume control in Abaqus.

"""

# -*- coding: mbcs -*-
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from mesh import *
from job import *
from sketch import *
from visualization import *
from regionToolset import Region

from abaqusConstants import ELLIPSE, ARC

from math import radians, exp, pi
import numpy as np

def getInitialVolume(radius, thickness, aspect, delta, sigma):
	# generate ellipsoidal shape of the midline
	nbPoints = 200
	theta = np.linspace(-np.pi/2, np.pi/2, nbPoints)
	rMid = radius*np.cos(theta)
	yMid = aspect*radius*np.sin(theta)

	# generate ellipsoidal shape of the inner surface
	rTangent = np.hstack(([1,], [rMid[i+1]-rMid[i-1] for i in range(1,len(rMid)-1)], [-1,]))
	yTangent = np.hstack(([0,], [yMid[i+1]-yMid[i-1] for i in range(1,len(yMid)-1)], [0,]))
	norms = np.array([np.sqrt(r**2+y**2) for r,y in zip(rTangent, yTangent)])
	rTangent = rTangent/norms
	yTangent = yTangent/norms

	rInEll = rMid - .5*thickness*yTangent
	yInEll = yMid + .5*thickness*rTangent

	# apply dimple imperfection to the inner surface
	angleWithAxis = np.arctan2(rInEll, yInEll)
	dimpleAmpl = -delta*np.exp(-.5*(angleWithAxis/sigma)**2)
	magn = np.sqrt(rInEll**2 + yInEll**2)

	rIn = rInEll + dimpleAmpl*rInEll/magn
	yIn = yInEll + dimpleAmpl*yInEll/magn

	# numerically calculate internal volume
	v = 0
	for i in range(nbPoints-1):
		rAv = .5*(rIn[i] + rIn[i+1])
		Dy = yIn[i+1] - yIn[i]
		v += np.pi*rAv**2*Dy

	return v

def generatePartGeometry(modeldb, prt, radius, thickness, eccentricity, aspect, elementType, dimensionality):
	rOut = radius + .5 * thickness
	offset = eccentricity * thickness

	# Setting up sketch
	skt = modeldb.ConstrainedSketch(name='__profile__', sheetSize=4*rOut)
	skt.sketchOptions.setValues(decimalPlaces=3, viewStyle=AXISYM)
	axis = skt.ConstructionLine(point1=(0.0, -2*rOut), point2=(0.0, 2*rOut))
	skt.FixedConstraint(entity=axis)
	oldKeys = set(skt.geometry.keys())

	# Drawing center profile
	prof = skt.EllipseByCenterPerimeter(center=(0.0, 0.0),
										axisPoint1=(0.0, aspect*radius),
										axisPoint2=(radius, 0.0))
	skt.autoTrimCurve(curve1=prof, point1=(-radius, 0.0))
	prof = skt.geometry[[k for k in skt.geometry.keys() if k not in oldKeys][0]]
	oldKeys = set(skt.geometry.keys())

	if elementType == 'solid':
		# Inner surface
		skt.offset(objectList=(prof,), distance=.5*thickness, side=LEFT)
		inner = skt.geometry[[k for k in skt.geometry.keys() if k not in oldKeys][0]]
		oldKeys = set(skt.geometry.keys())
		# Outer surface
		skt.offset(objectList=(prof,), distance=.5*thickness, side=RIGHT)
		outer = skt.geometry[[k for k in skt.geometry.keys() if k not in oldKeys][0]]
		outerVerts = outer.getVertices()
		# Profile to construction line
		skt.setAsConstruction(objectList=(prof,))
		# Apply inner surface eccentricity
		skt.move(objectList=(inner,), vector=(0.0, offset))
		innerVerts = inner.getVertices()
		# Top axis line
		innerTop = innerVerts[np.argmax([v.coords[1] for v in innerVerts])]
		outerTop = outerVerts[np.argmax([v.coords[1] for v in outerVerts])]
		topLine = skt.Line(point1=innerTop.coords, point2=outerTop.coords)
		# Bottom axis line
		innerBot = innerVerts[np.argmin([v.coords[1] for v in innerVerts])]
		outerBot = outerVerts[np.argmin([v.coords[1] for v in outerVerts])]
		botLine = skt.Line(point1=innerBot.coords, point2=outerBot.coords)
		# Consolidating part
		if dimensionality == '3D':
			prt.BaseSolidRevolve(sketch=skt, angle=180.0)
		else:
			prt.BaseShell(sketch=skt)
	else:
		if dimensionality == '3D':
			prt.BaseShellRevolve(sketch=skt, angle=180.0)
		else:
			prt.BaseWire(sketch=skt)

	del modeldb.sketches[skt.name]


def generatePartSets(modeldb, prt, radius, thickness, eccentricity, aspect, elementType, dimensionality):
	rxIn = radius - .5 * thickness if elementType == 'solid' else radius
	rxOut = radius + .5 * thickness if elementType == 'solid' else radius
	ryIn = aspect*radius - .5 * thickness if elementType == 'solid' else aspect * radius
	ryOut = aspect*radius + .5 * thickness if elementType == 'solid' else aspect * radius
	offset = eccentricity * thickness if elementType == 'solid' else 0.

	# Full part section
	prt.Set(name='Section', cells=prt.cells, faces=prt.faces, edges=prt.edges)

	# Creating inner surface
	if dimensionality == '3D':
		surfInt = part.FaceArray((prt.faces.getClosest(((0,offset,rxIn),))[0][0],))
		surfExt = part.FaceArray((prt.faces.getClosest(((0,0,rxOut),))[0][0],))
	else:
		surfInt = part.EdgeArray((prt.edges.getClosest(((rxIn,offset,0),))[0][0],))
		surfExt = part.EdgeArray((prt.edges.getClosest(((rxOut,0,0),))[0][0],))

	if elementType == 'solid':
		if dimensionality == '3D':
			prt.Surface(name='InnerSurface', side1Faces=surfInt)
			prt.Surface(name='OuterSurface', side1Faces=surfExt)
		else:
			prt.Surface(name='InnerSurface', side1Edges=surfInt)
			prt.Surface(name='OuterSurface', side1Edges=surfExt)
	else:
		if dimensionality == '3D':
			prt.Surface(name='InnerSurface', side1Faces=surfInt)
		else:
			prt.Surface(name='InnerSurface', side1Edges=surfInt)


		# Creating displacement point set
	poleVert = prt.vertices.getClosest(((0.,ryOut,0.),))[0][0]
	poleVertIn = prt.vertices.getClosest(((0.,ryIn,0.),))[0][0]
	antiVert = prt.vertices.getClosest(((0.,-ryOut,0.),))[0][0]
	antiVertIn = prt.vertices.getClosest(((0.,-ryIn,0.),))[0][0]
	prt.Set(name='Pole', vertices=part.VertexArray((poleVert,)))
	prt.Set(name='PoleInner', vertices=part.VertexArray((poleVertIn,)))
	prt.Set(name='Antipole', vertices=part.VertexArray((antiVert,)))
	prt.Set(name='AntipoleInner', vertices=part.VertexArray((antiVertIn,)))

		# Creating axis set
	if elementType == 'solid':
		topAxis = prt.edges.getClosest(((0,.5*(ryIn+offset+ryOut),0),))[0][0]
		bottomAxis = prt.edges.getClosest(((0,.5*(-ryIn+offset-ryOut),0),))[0][0]
		prt.Set(name='Axis', edges=part.EdgeArray((topAxis, bottomAxis)))
	else:
		prt.Set(name='Axis', vertices=part.VertexArray((poleVert, antiVert)))

		# Creating symmetry surface set
	if dimensionality == '3D':
		if elementType == 'solid':
			rightFace = prt.faces.getClosest(((radius, 0., 0.),))[0][0]
			leftFace = prt.faces.getClosest(((-radius, 0., 0.),))[0][0]
			prt.Set(name='Symmetry', faces=part.FaceArray((rightFace, leftFace)))
		else:
			rightEdge = prt.edges.getClosest(((radius, 0., 0.),))[0][0]
			leftEdge = prt.edges.getClosest(((-radius, 0., 0.),))[0][0]
			prt.Set(name='Symmetry', edges=part.EdgeArray((rightEdge, leftEdge)))


def generatePartMesh(prt, meshSize, elementType, dimensionality):
	# Partition to enable hex meshing
	if dimensionality == '3D':
		prt.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=0.)
		plane = prt.datums[max(prt.datums.keys())]
		if elementType == 'solid':
			prt.PartitionCellByDatumPlane(cells=prt.cells[0], datumPlane=plane)
		else:
			prt.PartitionFaceByDatumPlane(faces=prt.faces[0], datumPlane=plane)

	# Seed part
	prt.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=meshSize)
	# Set element type
	if elementType == 'solid':
		if dimensionality == '3D':
			#prt.setMeshControls(algorithm=MEDIAL_AXIS, regions=prt.cells)
			prt.setElementType(elemTypes=(
				ElemType(elemCode=C3D8RH, elemLibrary=STANDARD, kinematicSplit=AVERAGE_STRAIN, hourglassControl=DEFAULT),
				ElemType(elemCode=C3D6H, elemLibrary=STANDARD),
				ElemType(elemCode=C3D4H, elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)),
				regions=(prt.cells,))
		else:
			prt.setElementType(elemTypes=(
				ElemType(elemCode=CAX4H, elemLibrary=STANDARD, hourglassControl=DEFAULT),
				ElemType(elemCode=CAX3H, elemLibrary=STANDARD)),
				regions=(prt.faces,))
	else:
		if dimensionality == '3D':
			#prt.setMeshControls(algorithm=MEDIAL_AXIS, regions=prt.faces)
			prt.setElementType(elemTypes=(
				ElemType(elemCode=S4R, elemLibrary=STANDARD, secondOrderAccuracy=OFF, hourglassControl=DEFAULT),
				ElemType(elemCode=S3, elemLibrary=STANDARD, secondOrderAccuracy=OFF)),
				regions=(prt.faces,))
		else:
			prt.setElementType(elemTypes=(
				ElemType(elemCode=SAX1, elemLibrary=STANDARD, secondOrderAccuracy=OFF), ), 
				regions=(prt.edges,))
	# Create mesh
	prt.generateMesh()


def generatePartDimple(prt, magn, sigma, radius, thickness, dimensionality):
	# The imperfection consists of a gaussian dimple at the pole
	newCoordList = []
	dimpleAspect = 2. if dimensionality == '3D' else 1.
	for partNode in prt.nodes:
		# get current node coordinates
		nodeCoords = partNode.coordinates
		# get the shape of the deformation field
		r = sqrt(nodeCoords[0]**2+nodeCoords[1]**2+nodeCoords[2]**2)
		angleWithY = atan2(sqrt(nodeCoords[0]**2+nodeCoords[2]**2), nodeCoords[1])
		angleWithX = atan2(nodeCoords[2], nodeCoords[0])
		# get deformation field
		sigmaR = sigma #/ sqrt((1/dimpleAspect**2-1)*sin(angleWithX)**2+1)
		radialDeformation = -magn*exp(-.5*(angleWithY/sigmaR)**2)

		#radialDeformation = radialDeformation + .5*magn*exp(-.5*((angleWithY-sigmaR)/sigmaR)**2)

		# angleWithYX = atan2(nodeCoords[0], nodeCoords[1])
		# angleWithYZ = atan2(nodeCoords[2], nodeCoords[1])
		# tDimple = -magn*min(exp(angleWithY/sigma), exp(-angleWithY/sigma))*exp(-(angleWithYZ/(dimpleAspect*sigma))**2)

		# rIn = radius - .5*thickness
		# radialDeformation = tDimple*(r-rIn)/thickness

		# sigmaR = (radius+.5*thickness)*sin(min(sigma,pi/2))
		# rNorm = sqrt(dimpleAspect**2*nodeCoords[0]**2 + nodeCoords[2]**2)/sigmaR
		# radialDeformation = -magn*.5*(1+cos(rNorm*pi)) if rNorm <= 1. and nodeCoords[1] > 0 else 0

		dimpleDeformation = (radialDeformation*nodeCoords[0]/r,
 							 radialDeformation*nodeCoords[1]/r,
							 radialDeformation*nodeCoords[2]/r)

		# hingeDefMagn = .5*magn*exp(-.5*((angleWithY-radians(180))/sigmaR)**2)
		# hingeDeformation = (hingeDefMagn*nodeCoords[0]/r,
		# 					hingeDefMagn*nodeCoords[1]/r,
		# 					hingeDefMagn*nodeCoords[2]/r)

		# generate new node coordinates
		newCoordList.append([nodeCoords[i] \
							 + dimpleDeformation[i] for i in range(3)])

	# update the coordinate of the nodes in the mesh
	prt.editNode(nodes=prt.nodes, coordinates=newCoordList)


def generatePartSection(modeldb, prt, material, radius, thickness, eccentricity, elementType):
	# Creating section
	if elementType == 'solid':
		modeldb.HomogeneousSolidSection(material=material.name, \
										name='MatSection', thickness=None)
	else:
		prt.DatumCsysByThreePoints(name='Cylindrical', coordSysType=CYLINDRICAL,
								   origin=(0.,0.,0.), line1=(1.,0.,0.), line2=(0.,1.,0.))
		axis = [d for d in prt.datums.values() if hasattr(d, 'coordSysType') and d.coordSysType==CYLINDRICAL][0]
		if eccentricity == 0.:
			expr = '{t} + 0*Th'.format(t=thickness)
		else:
			expr = '{R}*(1+{t}/2) - {e}*{t}*{R}*(cos(Th-pi/2)+sqrt(pow((2-{t})/(2*{e}*{t}),2)-pow(sin(Th-pi/2),2)))'.format(R=radius, t=thickness, e=eccentricity)
		thicknessField = modeldb.ExpressionField(name='tField', localCsys=axis, expression=expr, description='')
		modeldb.HomogeneousShellSection(material=material.name,
				name='MatSection', thicknessType=NODAL_ANALYTICAL_FIELD,
				nodalThicknessField='tField', thickness=0.)

	# Section assignment
	if elementType == 'solid':
		prt.SectionAssignment(region=prt.sets['Section'], 
							  sectionName='MatSection')
	else:
		prt.SectionAssignment(region=prt.sets['Section'],
							  sectionName='MatSection',
							  offset=0., offsetType=MIDDLE_SURFACE)


def generateSphereModel(modeldb, params, it=0): 

	# ---------------------------------------------------
	# Initializing control parameters for this simulation
	# ---------------------------------------------------

	element_type = params.get('element_type', it)
	dimensionality = params.get('dimensionality', it)
	clamping = params.get('clamping', it)
		# Geometric control parameters
	R_s = 1.
	t_s = params.get('eta',it)
		# Imperfection control parameters
	magn = params.get('delta', it)
	sig = params.get('sigma', it)
	ecc = params.get('ecc', it)
	asp = params.get('alpha', it)
	asp = 1-1e-5 if asp == 1. else asp
		# Material control parameters
	mu = 1.
		# Load control parameters
	load_max = params.get('load_max', it)
		# Other control parameters
	it_max = params.get('it_max',it)
	step_max = params.get('step_max',it)
	mesh_size = params.get('mesh_size',it)


	# --------------------------
	# Defining geometry of shell
	# --------------------------

	if dimensionality == '3D':
		shell = modeldb.Part(name='Shell', dimensionality=THREE_D, 
							 type=DEFORMABLE_BODY)
	else:
		shell = modeldb.Part(name='Shell', dimensionality=AXISYMMETRIC, 
							 type=DEFORMABLE_BODY)
	generatePartGeometry(modeldb, shell, R_s, t_s, ecc, asp, element_type, dimensionality)
	generatePartSets(modeldb, shell, R_s, t_s, ecc, asp, element_type, dimensionality)

	# -------------------------------
	# Creating and assigning material
	# -------------------------------

		# Creating material
	mat = modeldb.Material(name='NeoHooke')
	mat.Density(table=((.5*mu/R_s**2*2e-7,),))
	mat.Damping(alpha=0.002, beta=0.002)
	mat.Hyperelastic(materialType=ISOTROPIC, table=((.5*mu, 0.0), ),
					 type=NEO_HOOKE, testData=OFF,
					 volumetricResponse=VOLUMETRIC_DATA)

	generatePartSection(modeldb, shell, mat, R_s, t_s, ecc, element_type)

	# -------------
	# Meshing parts
	# -------------

	generatePartMesh(shell, mesh_size, element_type, dimensionality)
	generatePartDimple(shell, magn, sig, R_s, t_s, dimensionality)

	# ------------------------------------
	# Creating assembly and extra features
	# ------------------------------------

		# Creating assembly
	ass = modeldb.rootAssembly
	ass.DatumCsysByThreePoints(coordSysType=CYLINDRICAL, origin=(0.0, 0.0, 0.0), \
							   point1=(1.0, 0.0, 0.0), point2=(0.0, 0.0, -1.0))
	shell = ass.Instance(dependent=ON, name='Shell-1', part=shell)

		# Creating cavity point
	cavityPoint = ass.ReferencePoint(point=(0.0, 0.0, 0.0))
	shellCRPName = 'ShellCavityRP'
	shellCRP = ass.Set(name=shellCRPName,
					   referencePoints=(ass.referencePoints[cavityPoint.id],))

	if clamping == 'average':
		# Creating constraint point
		constraintPoint = ass.ReferencePoint(point=(0.0, 0.0, 0.0))
		constraintPt = ass.Set(name='ConstrainRP',
							   referencePoints=(ass.referencePoints[constraintPoint.id],))

	# ------------------------------
	# Defining contents of the steps
	# ------------------------------

		# Creating step
	stepTime = 10.
	modeldb.ImplicitDynamicsStep(name='Inflation', previous='Initial', 
								 timePeriod=stepTime, maxNumInc=it_max, 
								 application=MODERATE_DISSIPATION, 
								 initialInc=step_max, minInc=1e-7, 
								 maxInc=step_max*stepTime/10., nlgeom=ON)

		# Creating fluid cavity properties
	shellCProp = modeldb.FluidCavityProperty(name='CavityProperty',
											 definition=HYDRAULIC,
											 fluidDensity=1.)
		# Creating fluid cavity interactions
	shellCavity= modeldb.FluidCavity(name='ShellCavity',
									 createStepName='Initial', 
									 cavityPoint=shellCRP, 
									 cavitySurface=shell.surfaces['InnerSurface'], 
									 interactionProperty=shellCProp.name)

		# Creating self contact interaction and its property
	contactProp = modeldb.ContactProperty('ContactProperty')
	contactProp.NormalBehavior(pressureOverclosure=EXPONENTIAL, 
							   table=((2*mu, 0.0), (0.0, 1e-2*t_s)))
	contactProp.TangentialBehavior(formulation=PENALTY, table=((1.15, ), ),
								   fraction=0.005, maximumElasticSlip=FRACTION)
	contactInt = modeldb.SelfContactStd(name='SelfContactInt', createStepName='Initial',
										surface=shell.surfaces['InnerSurface'], 
										interactionProperty=contactProp.name)
	if element_type == 'solid':
		contactExt = modeldb.SelfContactStd(name='SelfContactExt', createStepName='Initial',
											surface=shell.surfaces['OuterSurface'],
											interactionProperty=contactProp.name)

		# Creating coupling constraint
	if clamping == 'average':
		if dimensionality == '3D':
			modeldb.Coupling(name='CoupleSymmetry', controlPoint=constraintPt,
							 surface=shell.sets['Symmetry'], couplingType=DISTRIBUTING,
							 influenceRadius=WHOLE_SURFACE, weightingMethod=UNIFORM,
							 u1=ON, u2=ON, u3=OFF, ur1=OFF, ur2=OFF, ur3=ON)
		else:
			modeldb.Coupling(name='CoupleAxis', controlPoint=constraintPt, 
							 surface=shell.sets['Axis'], couplingType=DISTRIBUTING, 
							 influenceRadius=WHOLE_SURFACE, weightingMethod=UNIFORM,
							 u1=ON, u2=ON, ur3=OFF)


		# Creating history outputs
			# Pressure and volume
	del modeldb.historyOutputRequests['H-Output-1']
	modeldb.HistoryOutputRequest(name='PV', createStepName='Inflation', 
								 frequency=1, region=shellCRP, 
								 variables=('PCAV', 'CVOL'))
			# Detecting contact
	modeldb.HistoryOutputRequest(name='Contact', createStepName='Inflation',
								 frequency=1, interactions=('SelfContactInt', ),
								 variables=('CAREA', 'XN1', 'XN2'))

			# Displacements
	modeldb.HistoryOutputRequest(name='Pole', createStepName='Inflation', 
								 frequency=1, region=shell.sets['Pole'], 
								 variables=('U2',))
	modeldb.HistoryOutputRequest(name='Antipole', createStepName='Inflation', 
								 frequency=1, region=shell.sets['Antipole'], 
								 variables=('U2',))
	# This one is just a dummy to make sure there is a sample at exactly 
		# the midpoint of the simulation so inflation and deflation are
		# separated nicely
	modeldb.TimePoint(name='midPoint', points=((.5*stepTime,),))
	modeldb.HistoryOutputRequest(name='Dummy', createStepName='Inflation', 
								 timePoint='midPoint', region=shell.sets['Pole'], 
								 variables=('U1',))
	
			# Editing field outputs
	modeldb.fieldOutputRequests['F-Output-1'].setValues(frequency=1)
	if element_type == 'shell':
		modeldb.FieldOutputRequest(name='F-Output-2', frequency=1,
								   createStepName='Inflation', 
								   region=shell.sets['Section'],
								   variables=('STH', 'COORD'))


		# Creating boundary conditions
	modeldb.DisplacementBC(name='ConstrainX', createStepName='Initial',
						   region=shell.sets['Axis'], u1=SET, u2=UNSET, ur3=SET)
	if clamping == 'average':
		modeldb.DisplacementBC(name='ConstrainY', createStepName='Initial',
							   region=constraintPt, u1=SET, u2=SET, u3=SET,
							   ur1=SET, ur2=SET, ur3=SET)
	elif clamping == 'pin':
		modeldb.DisplacementBC(name='ConstrainY', createStepName='Initial',
							   region=shell.sets['Antipole'], u1=SET, u2=SET,
							   u3=SET, ur1=SET, ur2=SET, ur3=SET)
	if dimensionality == '3D':
		modeldb.DisplacementBC(name='ConstrainZ', createStepName='Initial',
							   region=shell.sets['Symmetry'], u3=SET, ur1=SET, ur2=SET)


		# Creating fluid flux amplitude
	accTime = 0.01
	amp = modeldb.TabularAmplitude(name='fluxAmplitude', timeSpan=STEP,
								   data=((0.,0.),
										 (accTime,1.),
										 (.5*stepTime-accTime,1.),
										 (.5*stepTime+accTime,-1.),
										 (stepTime-accTime,-1.),
										 (stepTime,0.)))
	ampIntegral = .5*stepTime - accTime
	V0 = getInitialVolume(R_s, t_s, asp, magn, sig) #4./3.*pi*asp*(R_s-.5*t_s)**3
	if dimensionality == '3D':
		V0 *= .5

		# Creating fluid flux load
	modeldb.keywordBlock.synchVersions()
	# find keyword block right before the end of the definition of the
	# inflation step
	i = 0
	while not modeldb.keywordBlock.sieBlocks[i].lower().startswith("*step, name=inflation"):
		i += 1
	while not modeldb.keywordBlock.sieBlocks[i].lower().strip() == "*end step":
		i += 1
	modeldb.keywordBlock.insert(i-1, '*Fluid flux, AMP={}\n{}, {}'.format(
								amp.name, shellCRPName,
								-V0*load_max/ampIntegral))

	# --------------------------
	# Job creation and execution
	# --------------------------
	
	# Creating job
	job = mdb.Job(model=modeldb.name, name=modeldb.name, type=ANALYSIS, 
				  numCpus=1, numDomains=1)

	return job