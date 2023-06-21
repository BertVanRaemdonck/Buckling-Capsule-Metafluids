# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:10:00 2020

@author: u0123347
"""

from numpy import ndarray


class controlParameters:
	"""
	Get values from a dictionary with parameters by index

	Takes a dictionary with both vectors and scalars as an
	input. Scalars are treated as a shorthand for vectors
	with the same entry for every element. Then allows to
	get all parameters by index
	"""
	def __init__(self, param_dict):
		self.parameters = param_dict

	def get(self, name, index=0):
		"""
		Get the value of the given parameter at the index

		This value is the element at the given index in a
		vector or just the scalar value if no vector was
		provided for that parameter
		"""
		parameterContent = self.parameters[name]
		if isinstance(parameterContent, (list, tuple, ndarray)):
			return parameterContent[min(index, len(parameterContent)-1)]
		return parameterContent

	def getAll(self, index=0):
		"""
		Get dictionary with all properties corresponding to
		the index
		"""
		return {k:self.get(k,index) for k in self.parameters.keys()}

	def getNbSimulations(self):
		"""
		Get the amount of distinct entries contained in the
		parameters.

		This equals the maximum possible index that can be
		supplied to the get functions
		"""
		nbSimulations = 1
		for parameterKey in self.parameters.keys():
			parameterContent = self.parameters[parameterKey]
			if isinstance(parameterContent, (list, tuple, ndarray)):
				nbSimulations = max(nbSimulations, len(parameterContent))
		return nbSimulations