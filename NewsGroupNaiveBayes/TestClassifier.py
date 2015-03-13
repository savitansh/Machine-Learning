from NaiveBayesClassifier import *
stopWords = ['on','the']
t=NaiveBayes("news_data/20news-bydate-train/" , stopWords)
t.test("news_data/20news-bydate-test/")
