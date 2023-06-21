Main script
-----------

- ManageSphereSimulations.py
	This script runs a series of simulations in Abaqus with different parameters, does some processing on the results, and creates a json file for each simulation containing the simulation settings and output. To use this script, edit it in a text editor with the desired simulation parameters, then run it in Abaqus CAE by going to File > Run script and navigating to this folder.

Dedicated helper scripts
------------------------
- CreateSphereSimulation.py
	Script called by ManageSphereSimulations.py to create a simulation with a given set of parameters
- ProcessSphereSimulation.py
	Script called by ProcessSphereSimulation.py after a simulation has finished to extract the relevant output from the simulation
	
General helper scripts
----------------------

See the Dependencies folder