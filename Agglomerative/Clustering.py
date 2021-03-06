import os, codecs, math

class Agglomerative:
	def __init__(self, trainingdir):
		"""Find tfidf scores of docs and 
		make a cosine simmilarity matrix"""

		categories = os.listdir(trainingdir)
		#filter out files that are not directories
		self.categories = [filename for filename in categories if os.path.isdir(trainingdir + filename)]
		
		self.vocabulary = {}
		self.topFrequentWords = {}
		self.idfscores = {}
		self.tfscores = {}

		numberofdocs = 0
		label_count=0
		self.labels = {}
		for category in self.categories:
			files = os.listdir(trainingdir + category)
			label_count = label_count+1
			for f in files:
				self.labels[f] = label_count
				f = codecs.open(trainingdir + category + "/" + f, 'r')
				for line in f:
					tokens = line.split()
					for token in tokens:
						token = token.strip('\'".,?:-')
						token = token.lower()
						self.vocabulary.setdefault(token, 0)
						self.vocabulary[token] += 1

				numberofdocs += 1
		results ={}
		results = list(self.vocabulary.items())
		results.sort(key=lambda tuple: tuple[1], reverse = True)

		self.topFrequentWords = results[:1000]
		wordsdict = {}
		for entry in self.topFrequentWords:
			key = entry[0]
			wordsdict[key] = entry[1]

		

		docCount ={}
		self.docids = []		
		tempdict ={}
		docLen = {}

		for category in self.categories:
			files = os.listdir(trainingdir + category)
			for f in files:
				docid = str(f)
				docLen.setdefault(docid , 0)

				self.docids.extend([docid]) 
				docmap = {}
				tf = {}
				tempdict = wordsdict.copy()
				f = codecs.open(trainingdir + category + "/" + f, 'r')
				for line in f:
					tokens = line.split()
					for token in tokens:
						token = token.strip('\'".,?:-')
						token = token.lower()
						docLen[docid] += 1
						if token in tempdict:
							docCount.setdefault(token,0)
							docCount[token] += 1
							docmap[token] = 1
							del tempdict[token]
						if token in wordsdict:
							tf.setdefault(token,0)
							tf[token] += 1
									
				self.idfscores[docid] = docmap	
				self.tfscores[docid] = tf			
		
		self.tfidfscores = {}

		for docid in self.docids:
			docmap = self.idfscores[docid].copy()

			for word in docmap:
				self.idfscores[docid][word] = math.log(float(numberofdocs / float(docCount[word])))
				docmap[word] = self.idfscores[docid][word] * self.tfscores[docid][word]

			self.tfidfscores[docid] = docmap

		self.simmilaritymatrix = {}
		
		l = len(self.docids)

		for i in range(0,l):
			simmVec = {}
			did1 = self.docids[i]
			for j in range(0,l):
				did2 = self.docids[j]
				simmVec[did2] = float(self.simmilarity(did1 , did2) / (docLen[did1] * docLen[did2]))
			self.simmilaritymatrix[did1] = simmVec	 
		
		self.makecluster()

	def simmilarity(self , docid1 , docid2):
		"""Find cosine simmilarity distance
		between two given documents"""

		dic1 = self.tfidfscores[docid1]
		dic2 = self.tfidfscores[docid2]
		
		val = 0
		for word in dic1:
			val += (dic1[word] * dic1[word])

		mag1 = math.sqrt(val)
		
		val = 0
		for word in dic2:
			val += (dic2[word] * dic2[word])

		mag2 = math.sqrt(val)


		cummulativeProd = 0
		for word in dic1:
			if word in dic2:
				cummulativeProd += (dic1[word] * dic2[word])


		simmdist = float(cummulativeProd / mag1 * mag2)
		return simmdist				

	def makecluster(self):
		"""Make clusters from given docids"""
		self.clusterList = []
		for did in self.docids:
			cluster = []
			cluster = cluster + [did]
			self.clusterList = self.clusterList + [cluster]
		
		l = len(self.clusterList)

		while l > 30:
			mindist = 100000000

			s = 0
			for i in range(0,l):
				for j in range(i+1,l):
					s = self.clusterDistanceType2( i , j)
					if s < mindist:
						mindist = s
						clustindx1 = i
						clustindx2 = j
			self.clusterList[clustindx1] = self.clusterList[clustindx1] + self.clusterList[clustindx2]
			t = self.clusterList[clustindx2]
			self.clusterList.remove( t )
			l = len(self.clusterList)
			
		print self.clusterList	


	def clusterDistanceType1(self ,cluster1 , cluster2):
		
		minDist = 1000000
		docids1 = self.clusterList[cluster1]
		docids2 = self.clusterList[cluster2]

		for x in docids1:
			for y in docids2:
				if x != y:
					if self.simmilaritymatrix[x][y] > 0:
						d = self.simmilaritymatrix[x][y]
						if d < minDist and d > 0:
							minDist = d
		return minDist				


	def clusterDistanceType2(self ,cluster1 , cluster2):
		
		maxDist = 0
		docids1 = self.clusterList[cluster1]
		docids2 = self.clusterList[cluster2]

		for x in docids1:
			for y in docids2:
				if x != y:
					if self.simmilaritymatrix[x][y] > 0:
						d = self.simmilaritymatrix[x][y]
						if d > maxDist and d > 0:
							maxDist = d
		return maxDist

	def clusterDistanceType3(self ,cluster1 , cluster2):
		
		avgDist = 0
		docids1 = self.clusterList[cluster1]
		docids2 = self.clusterList[cluster2]
		totalDist = 0
		numOfSamples = len(docids1) + len(docids2)

		for x in docids1:
			for y in docids2:
				if x != y:
					if self.simmilaritymatrix[x][y] > 0:
						d = self.simmilaritymatrix[x][y]
						totalDist = totalDist + d
		
		avgDist = float(totalDist / numOfSamples)
		return avgDist						
