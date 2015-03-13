import os, codecs, math

class Classifier:
	def __init__(self, trainingdir , testdir , knntraindir , stopwordsdir):
		"""Find tfidf scores of docs and 
		make a cosine simmilarity matrix"""
		f = codecs.open(stopwordsdir, 'r')

		stopwords = {} 
		for line in f:
			line = line[:-1]
			stopwords[line] = 1

		
		categories = os.listdir(trainingdir)
		#filter out files that are not directories
		self.categories = [filename for filename in categories if os.path.isdir(trainingdir + filename)]
		
		self.vocabulary = {}
		self.topFrequentWords = {}
		self.idfscores = {}
		self.tfscores = {}
		self.labels = {}
		numberofdocs = 0

		for category in self.categories:
			files = os.listdir(trainingdir + category)
			for f in files:
				self.labels[f] = category
				f = codecs.open(trainingdir + category + "/" + f, 'r')
				for line in f:
					tokens = line.split()
					for token in tokens:
						token = token.strip('\'".,?:-<>;:*&^%$#@')
						token = token.lower()
						self.vocabulary.setdefault(token, 0)
						if token not in stopwords:
							self.vocabulary[token] += 1

				numberofdocs += 1
		results ={}
		results = list(self.vocabulary.items())
		results.sort(key=lambda tuple: tuple[1], reverse = True)

		self.topFrequentWords = results[:9000]
		# for word in self.topFrequentWords:
		# 	print word

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
				docLen.setdefault(f , 0)
				docid = f
				self.docids.extend([f]) 
				docmap = {}
				tf = {}
				tempdict = wordsdict.copy()
				f = codecs.open(trainingdir + category + "/" + f, 'r')
				for line in f:
					tokens = line.split()
					for token in tokens:
						token = token.strip('\'".,?:-<>;:*&^%$#@')
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
		

		self.testdata = {}
		files = os.listdir(testdir)
		for f in files:
			self.testdata.setdefault(f,0)

		self.trainingdata = []
		for category in self.categories:
			files = os.listdir(knntraindir + category)
			for f in files:
				self.trainingdata = self.trainingdata + [f]	

		self.classify(self.testdata , self.trainingdata , 35)	

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


	def classify(self , testdata , traininddata , k):
		matchcount = 0
		n_samples = len(testdata)
		for f in self.testdata:
			datadist = {}
			labelcount = {}
			for did in self.trainingdata:
				d = self.simmilaritymatrix[f][did]
				datadist[did] = d
			results ={}
			results = list(datadist.items())
			results.sort(key=lambda tuple: tuple[1], reverse = True)

			knn = {}
			knn = results[:k]

			for i in knn:
				label = self.labels[i[0]]
				labelcount.setdefault(label,0)
				labelcount[label] += 1
				#print label,i[0],":",self.labels[f],f
			results ={}
			results = list(labelcount.items())
			results.sort(key=lambda tuple: tuple[1], reverse = True)

			majoritylabel = results[0][0]

			self.testdata[f] = majoritylabel
			if self.testdata[f] == self.labels[f]:
				matchcount += 1
			#print f , self.testdata[f] , self.labels[f]	
		print matchcount , n_samples	
	
