import numpy as np 
import scipy.interpolate as interpolate
from typing import Tuple, Callable

class DistributionMap():
	"""
	Class for creating distribution map: X -> Y
	"""

	def __init__(self, X:np.array, Y:np.array, n_bins:int=10):
		"""
		Args:
			:X: (np.array) -> Array with distribution X, argument of map
			:Y: (np.array) -> Array with distribution Y, argument of map
			:n_bins: (int) How many bins use to create map
		"""

		self.bin_edges_X, self.cum_values_X = self.get_edges_values(X, n_bins)
		self.bin_edges_Y, self.cum_values_Y = self.get_edges_values(Y, n_bins)

		self.map = self.make_map(**self.__getstate__())

	def get_edges_values(self, data:np.array, n_bins:int) -> Tuple[np.array]:
		"""
		Returns bin's edges with their values.
		Args:
			:data: (np.array) Input data to getting map.

		return: bin_edges, cum_values
		"""

		hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
		cum_values = np.zeros(bin_edges.shape)
		cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))

		return bin_edges, cum_values

	def make_map(self, bin_edges_X:np.array, cum_values_X:np.array, bin_edges_Y:np.array, cum_values_Y:np.array) -> Callable:
		"""
		Making map φ: bin_edges_X -> bin_edges_Y
		Args:
			:bin_edges_X: (np.array) -> Bin's edges from X data
			:bin_edges_Y: (np.array) -> Bin's edges from Y data
			:cum_values_X: (np.array) -> Accumulated values from X data
			:cum_values_Y: (np.array) -> Accumulated values from X data

		return: lambda function bin_edges_X -> bin_edges_Y.
		"""

		#Map: bin_edges_X -> cum_values_X
		f_1 = interpolate.interp1d([0, *list(bin_edges_X), 1], [-np.inf, *list(cum_values_X), np.inf])

		#Map: cum_values_Y -> bin_edges_Y
		f_2 = interpolate.interp1d([-np.inf, *list(cum_values_Y), np.inf], [0, *list(bin_edges_Y), 1])

		return lambda x: f_2(f_1(x))

	def __call__(self, x):
		return self.map(x)

	def __getstate__(self) -> dict:
		"""
		Magic method for pickle dump.
		"""
		state = {
		'bin_edges_X': self.bin_edges_X,
		'bin_edges_Y': self.bin_edges_Y,
		'cum_values_X': self.cum_values_X,
		'cum_values_Y': self.cum_values_X,
		}

		return state 


	def __setstate__(self, state) -> None:
		"""
		Magic method for pickle load.
		"""
		for key, value in state.items():
			self.__setattr__(key, value)

		self.map = self.make_map(**state)

		return None

