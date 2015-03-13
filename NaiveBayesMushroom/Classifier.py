import os, codecs, math

class NaiveBayes:
	def __init__(self, trainingdir , featuresetdir , testdir):
		probp = []
		probe = []
		f = codecs.open(featuresetdir, 'r')
		for line in f:
			d1 = {}
			d2 = {}
			l = len(line)
			for i in range(0,l):
				if line[i] == '=':
					val = line[i+1]
					d1.setdefault(val,0)
					d2.setdefault(val,0)
			probp = probp + [d1]
			probe = probe + [d2]
		
		f = codecs.open(trainingdir, 'r')

		for line in f:
			tokens = line.split(",")
			label = tokens.pop(0)
			i = 0
			for token in tokens:
				token = token.strip("\n")
				if label == 'p':
					probp[i][token] += 1
					#print "pos",probp[i][token]
				else:
					probe[i][token] += 1
				i += 1		
		
		totalp = []
		l = len(probp)
		n_poisonous = 0.0
		n_edible = 0.0
		for token in probp[0]:
			n_poisonous +=  probp[0][token]
		
		for token in probe[0]:
			n_edible +=  probe[0][token]
			
		n_samples = n_poisonous + n_edible
			
		for i in range(0,l):
			for token in probp[i]:
				probp[i][token] = float(float(probp[i][token]) / float(n_poisonous))
				probe[i][token] = float(float(probe[i][token]) / float(n_edible))

		classPriorPoisonous = float(n_poisonous / n_samples)
		classPriorEdible = float(n_edible / n_samples)
		


		f = codecs.open(testdir, 'r')

		classifiedLabels = []
		groundtruth = []
		n_test_samples = 0
		for line in f:
			n_test_samples += 1
			tokens = line.split(",")
			label = tokens.pop(0)
			# Ground truth contains label obtained from the test_data file.
			groundtruth = groundtruth + [label]
			i = 0
			hp = 1.0
			he = 1.0
			for token in tokens:
				token = token.strip("\n")
				hp = float(hp *	 probp[i][token])
				he = float(he * probe[i][token])				
				i += 1
			hp = hp * classPriorPoisonous
			he = he * classPriorEdible

			# assign label to test sample based on which class has higher probability
			if( hp > he):
				classifiedLabels = classifiedLabels + ['p']
			else:
				classifiedLabels = classifiedLabels + ['e']

		hitCount = 0.0		
		for i in range(0,n_test_samples):
			if classifiedLabels[i] == groundtruth[i]:
				hitCount += 1	
		
		accuracy = float(hitCount / float(n_test_samples))

		print "accuracy = " , str(accuracy * 100 ), "%"



