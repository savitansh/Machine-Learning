"""
=========================================================
1NN classifier on lung cancer dataset
=========================================================

Show below is a basic 1nn classifier on lung cancer data which prints
 accuracy, confusion matrix
"""
print(__doc__)


import numpy as np
from numpy import genfromtxt
from numpy import matrix
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class KNN_Classifier:

	def __init__(self, training_dir, test_dir):
		"""make dataset and classify """

		f = open(training_dir,'r')
		sample = []
		self.X = []
		self.y = []
		for line in f:
			sample = line.split(",")
			s2 = sample[1:]
			X1 = []
			for s in s2:

				if s == "?":
					s = "1000"

				X1 = X1 + [float(s)]
			self.X = self.X + [X1]	
			self.y = self.y + [sample[0]]
		
		f = open(test_dir,'r')
		sample = []
		self.X_test = []
		self.y_test = []
		for line in f:
			sample = line.split(",")
			s2 = sample[1:]
			X1 = []
			for s in s2:
				if s == "?":
					s = "1000"
				X1 = X1 + [float(s)]
			self.X_test = self.X_test + [X1]
			self.y_test = self.y_test + [sample[0]]
		
		labels = self.classify()
		accuracy = self.get_accuracy(labels)
		self.confusion_mat = [[0,0,0],[0,0,0],[0,0,0]]
		self.gen_confusion_mat(labels)
		print "accurcy = ",accuracy
		print 
		print "confusion matrix : "
		for i in self.confusion_mat:
			print i
		
		
		
	def classify(self):
		i = 0
		label = []
		dis = 0
		for test_sample in self.X_test:
			min_dis = 10000
			j = 0
			indx = 0
			for training_sample in self.X:
				dis = self.find_distance(test_sample,training_sample)
				if(dis < min_dis):
					min_dis = dis
					indx = j
				j += 1
			label = label + [self.y[indx]]	
		return label	

	def find_distance(self, s1, s2):
		i = 0
		dis = 0
		for j in s1:
			v1 = s1[i]
			v2 = s2[i]
			dis += (v1 - v2) * (v1 - v2)
			i += 1
		dis = math.sqrt(dis)
		return dis	

	def get_accuracy(self, labels):
		i = 0
		match_count = 0.0
		accuracy = 0.0
		for label in self.y_test:
			if label == labels[i]:
				match_count += 1
			i += 1	
		accuracy = match_count / i
		return accuracy	

	def gen_confusion_mat(self, labels):
		M = {}
		M["1"] = 0
		M["2"] = 1
		M["3"] = 2

		i = 0
		for s1 in labels:
			s2 = self.y_test[i]
			if s1 != '' and s2 != '':
				marked_label = M[s1]
				true_label = M[s2]
				self.confusion_mat[ marked_label ][ true_label ] += 1
			i += 1
