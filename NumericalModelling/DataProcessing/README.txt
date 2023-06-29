Processing experimental and numerical results
=============================================

System requirements
-------------------

- Python 3 (tested on Python 3.9.13 on Windows)
- numpy
- scipy

Installation instructions
-------------------------

- All code has been written in python, so no special installation is required when python with the required dependencies has been installed on the system.
- Some filepaths may need to be updated (e.g. to point to the location where a file containing experimental data can be found on the system of the user)

Usage
-----

- All files of which the filename starts with Fig... are used to generate the figures in the main text. Therefore, these files are single use
- Processing of the provided simulation data is done using the `SphericalShell` class contained in `NumericalModelling/DataProcessing/Sphericalshell.py`. See the documentation of this class for a further explanation of its properties. A demonstration of the usage of this class can be found in `NumericalModelling/DataProcessing/Fig2FluidModelling.py` in the section "Load data"
- Modelling of suspensions of multiple capsules is done using the `PVCurve` class contained in `NumericalModelling/DataProcessing/Dependencies/PVCurveOperations.py`. A demonstration of the usage of this class can be found in `NumericalModelling/DataProcessing/Fig1CompareSimulationsExperimentsMultipleCapsules.py` in the section "Get simulations for multiple shells"
