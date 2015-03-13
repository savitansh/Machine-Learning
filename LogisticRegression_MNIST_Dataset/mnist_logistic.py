#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Logistic Regression 2-class Classifier
=========================================================

Show below is a logistic-regression classifiers decision boundaries on the
mnist dataset
displayed are accuracies for each pair of class
"""
print(__doc__)


import numpy as np
from numpy import genfromtxt
from numpy import matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

class Logistic:

	def __init__(self, training_dir, test_dir):
		""" generate model """	
		data = genfromtxt(training_dir, delimiter=',')  
		X = data[:, 1:]
		y = data[:, 0] 
		data = genfromtxt(test_dir, delimiter=',')  			
		X_test = data[:, 1:]
		self.y_test = data[:, 0] 

		h = .02  # step size in the mesh

		logreg = linear_model.LogisticRegression(C=1e5)

		class_type_train = {}
		class_type_test = {}
		class_sample_indx = {}
		self.class_label = {}

		for c in range(0,10):
			l = len(y)
			d = []
			for i in range(0,l):
				if(y[i] == c):
					d = d + [X[i]] 
			class_type_train[c] = d		

		self.vote_count = {}

		l = len(self.y_test)

		for i in range(0,l+1):
			v = {}
			for c in range(0,10):
				v[c] = 0
			self.vote_count[i] = v
				
		for c in range(0,10):
			d = []
			d2 = []
			for i in range(0,l):
				if(self.y_test[i] == c):
					d = d + [X_test[i]]
					d2 = d2 + [i+1] 
			class_type_test[c] = d		
			class_sample_indx[c] = d2


		
		acc = [0.0] * 10
		accuracy = [acc] * 10
		accuracies = []

		for c1 in range(0,10):
			for c2 in range(c1+1,10):
				hit_count = 0.0

				t = []
				t = class_type_train[c1] + class_type_train[c2]

				y1 = []
				for i in class_type_train[c1]:
					y1 = y1 + [c1]

				for i in class_type_train[c2]:
					y1 = y1 + [c2]
				l = len(t)
				X1 = t
				# Training model with c1 and c2 class types
				logreg.fit(X1, y1)

				# generating test dataset of c1 and c2 class types
				test_set = []
				ground_truth = []

				test_set = class_type_test[c1] + class_type_test[c2]
				test_sample_no = class_sample_indx[c1] + class_sample_indx[c2]
				for c in class_type_test[c1]:
					ground_truth += [c1]
				for c in class_type_test[c2]:
					ground_truth += [c2]
				
				n_samples = len(ground_truth)

				l_test = len(test_set)
				for j in range(0,l_test):
					z = logreg.predict(test_set[j])
					if z == ground_truth[j]:
						hit_count += 1	
					if z == c1:
						self.vote_count[test_sample_no[j]][c1] += 1
					else:
						self.vote_count[test_sample_no[j]][c2] += 1
					 	
				accuracy[c1][c2] = float(hit_count / n_samples)		
				accuracies += [accuracy[c1][c2]]

		print " sorted order of accuracies of 2-class classifiers= "
		print sorted(accuracies)
		self.voting_classify()		

	def voting_classify(self):
		""" find majority vote for each class from each model
		 and assign corresponding label """
		final_hit_count = 0.0	  
		l = len(self.y_test)
		for j in range(1,l+1):
			majority_count = 0
			majority_class = -1
			for c in range(0,10):
				if self.vote_count[j][c] > majority_count:
					majority_count = self.vote_count[j][c]
					majority_class = c
			self.class_label[j] = majority_class
			if self.y_test[j-1] == self.class_label[j]:
				final_hit_count += 1
				
		final_accuracy = float(final_hit_count / l)
		print "-------------------------------"
		print
		print " final accuracy after voting of all classifiers =",final_accuracy

			