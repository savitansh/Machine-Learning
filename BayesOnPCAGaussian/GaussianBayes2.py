"""
=======================================================
Perform Unimodal Gaussian bayes classification on
 MNIST PCA projected and unprojected data
=======================================================


Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data. Here we
plot the different samples on the 2 first principal components.

Unimodal gaussian with spherical shape gaussian over data without pca
gives 75 %  accuracy and with pca gives 12%  accuracy  
"""
print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import numpy as np
from numpy import genfromtxt
from numpy import matrix
from numpy.linalg import inv
from numpy.linalg import det
import math

class Classifier:

		def __init__(self, training_dir, test_dir):
			
			data = genfromtxt(training_dir, delimiter=',')
			labeled_test_data = genfromtxt(test_dir, delimiter=',')	
			X = data[:, 1:]
			y = data[:, 0] 
			test_data = labeled_test_data[:, 1:]
			ground_truth = labeled_test_data[:, 0]
			
			pca = PCA(n_components=9)
			X_r = pca.fit(X).transform(X)
			#lda = LDA(n_components=9)
			#X_r = lda.fit(X, y).transform(X)
			test_data = pca.fit(test_data).transform(test_data)

			
			class_means = {}
			self.n_dimensions = 9
			n_points = len(X_r)
			
			self.class_labels = {}

			for p in range(0,n_points):
				self.class_labels[y[p]] = 1
			
			training_set = {}
			self.class_prior = {}
			for c in self.class_labels:
				class_points = []
				size = 0.0
				for i in range(0,n_points):
					if y[i] == c:
						class_points = class_points + [X_r[i,:]]
						size += 1
				training_set[c] = class_points
				self.class_prior[c] = float(size / n_points)

				

			l = len(training_set)
			
			self.class_means = {}
			self.class_cvs = {}
			self.variance = 0.0

			for c in self.class_labels:
				training_data = training_set[c]
				training_data = np.array(training_data)
				
				(means, cv) = self.train_model(training_data)
				self.class_means[c] = means
				self.class_cvs[c] = cv

				
			#print self.class_means	
			#print self.class_cvs
			self.label = []	
					
			self.classify(test_data)	

			self.accuracy = 0.0
			self.evaluate(ground_truth)

			print "acc= ",self.accuracy * 100, " %"

#--------------------------------------------------------------------------------------------#			

		def train_model(self , training_data):
			means = [0.0] * self.n_dimensions

			n_points = len(training_data)

			for d in range(0,self.n_dimensions):
				t = 0.0
				for i in range(0, n_points):
					t += training_data[i][d]
				means[d] = float(t / n_points)

			for d in range(0,self.n_dimensions):
				val = 0.0
				for i in range(0, n_points):
					val = val + (training_data[i][d] - means[d]) * (training_data[i][d] - means[d])
				self.variance = max(self.variance, val) 	
				
			self.variance = self.variance / n_points
			self.variance = math.sqrt(self.variance)	
			cv = np.cov(training_data.T)
			#print cv
			return means, cv
					

#--------------------------------------------------------------------------------------------#

		def get_likelihood(self, point, mean, cv):

			# cv_inv = inv(cv)
			#cv_det = det(cv)
			
			delta = np.subtract(point, mean)
			max1 = 0

			
			for i in range(0,len(delta)):
				delta[i] = delta[i] / cv[i][i]	
			
			delta_T = delta.T
			v = delta
			v = np.dot(v, cv)
			v2 = math.pow(np.dot(v , delta_T) , 0.5)
			e1 = math.pow(2.303,-v2)
			likelihood = e1
			return likelihood	

#--------------------------------------------------------------------------------------------#

		def classify(self, test_data):
			n_pts = len(test_data)

			for p in range(0,n_pts):
				max_prob = 0.0
				prediction = 0
				for c in self.class_labels:
					prob = self.get_likelihood(test_data[p], self.class_means[c], self.class_cvs[c])
					prob = prob * self.class_prior[c]
					if prob > max_prob:
						max_prob = prob
						prediction = c
				self.label = self.label + [prediction]			

#--------------------------------------------------------------------------------------------#

		def evaluate(self, ground_truth):
			l = len(ground_truth)
			hit_count = 0.0
			for i in range(0,l):
				if ground_truth[i] == self.label[i]:
					hit_count += 1 				 
			
			self.accuracy = float(hit_count / l)		


