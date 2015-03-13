"""
=======================================================
Perform soft K means clustering on PCA projected MNIST dataset 
=======================================================

The Mnist dataset represents handwritten digits in 24x24 dimension pixel space

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data. Here we
plot the different samples on the 2 first principal components.

Soft k means clustering calculates soft degree of association of all data points
and then calculates updated means. This process is performed repeatedly till
the clusters converge till a certain level.
"""
print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import numpy as np
from numpy import genfromtxt
import os, codecs, math


class SoftClustering:
	def __init__(self ):

		data = genfromtxt('mnist_train_short.csv', delimiter=',')	
		X = data[:, 1:]
		y = data[:, 0] 

		target_names = ['0','1','2','3','4','5','6','7','8','9']

		pca = PCA(n_components=50)
		self.X_r = pca.fit(X).transform(X)


		# Percentage of variance explained for each components
		# print('explained variance ratio (first nine components): %s'
		#       % str(pca.explained_variance_ratio_))

		
		self.means = [] # To change

		self.means = self.means + [self.X_r[10]]
		self.means = self.means + [self.X_r[20]]
		self.means = self.means + [self.X_r[30]]
		self.means = self.means + [self.X_r[40]]
		lamda = 2
		
		dim = len(self.X_r[0])

		n_pts = len(self.X_r)

		m = [0] * dim
		self.delta_matrix = []

		
		for d in range(0,dim):
			val = 0
			for point in self.X_r:
				val += point[d]
			m[d] = float(val / n_pts)

		s = 0
		for d in range(0,dim):
			for point in self.X_r:
				s = s + (point[d] - m[d]) * (point[d] - m[d])

		avg_var = float(s / n_pts)		
		
		K = 4
		n_dimensions = dim
		d = []
		for j in range(0,K):
				d = d + [0.0]
			
		for i in range(0,n_pts):
			self.delta_matrix = self.delta_matrix + [d]

		threshold = 0.7	
				
		for iteration in range(0, 10):
			i = 0
			for point in self.X_r:
				k = 0

				self.association_extent = []
				for i in range(0,K):
					self.association_extent = self.association_extent + [0.0]

				self.find_association(point, avg_var , K)	
				
				#print self.association_extent
				self.delta_matrix[i] = self.association_extent		
				i += 1

			self.oldmeans = list(self.means)

			self.means = []
			association_list = []
			for k in range(0,K):
				i = 0
				total_association = 0.0
				for i in range(0,n_pts):
					total_association += self.delta_matrix[i][k]
				association_list = association_list + [total_association]
					
			for k in range(0,K):
				mean = []
				for d in range(0,n_dimensions):
					v = 0.0
					for p in range(0,n_pts):
						v = v + self.delta_matrix[p][k] * self.X_r[p][d]
					mean = mean + [float( v / association_list[k])]		
				self.means = self.means + [mean]
				#print mean
			avg_var = avg_var * lamda

			maxdiff = 0.0
			for k in range(0,K):
				for d in range(0,dim):
					diff = self.oldmeans[k][d] - self.means[k][d]
					if diff > maxdiff:
						maxdiff = diff

			print maxdiff			
			if maxdiff < threshold:
				break			
			print iteration	

	def find_association(self, point, avg_var, K):
		for k in range(0,K):
			l = len(point)
			s = 0
			for i in range(0,l):
				s += (point[i] - self.means[k][i]) * (point[i] - self.means[k][i])
			s = float(s / avg_var)
			s = math.exp(-s)
			self.association_extent[k] = s


