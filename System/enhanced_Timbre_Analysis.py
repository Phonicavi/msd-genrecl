# -*- coding:utf-8 -*-  
from BasicClass_eTA import *
import sys
import random
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.metrics import classification_report as clfr
from sklearn.svm import SVC
import math

NUM_NEED_PER_GENRE = [200,200,200,200]
GENRES = ['Jazz','Rap','Rock','Country']

def multi_SVM(needcv = False):
	NeedReFetch = False
	allGenreSongsTrain,allGenreSongsTest = fetchData_eTA(NUM_NEED_PER_GENRE,GENRES,NeedReFetch)

	# assert(len(allGenreSongsTrain[0][0]) == 106)

	TrainX = []
	TrainY = []
	TestX = []
	TestY = []
	for i in range(len(GENRES)):
		for j in allGenreSongsTrain[i]:
			TrainX.append(j)
			TrainY.append(i)
		for k in allGenreSongsTest[i]:
			TestX.append(k)
			TestY.append(i)
	confuseMat = [[0 for i in range(len(GENRES))] for j in range(len(GENRES))];
	if not needcv:
		print "Start SVM training ... "
		model = SVC()
		model.fit(TrainX,TrainY)
		print "Start SVM predicting ... "
		PredY = model.predict(TestX)
		for i in range(len(TestY)):
			confuseMat[TestY[i]][PredY[i]] += 1
	else:
		tuned_parameters = [															## remained to be play with
							{'kernel': ['rbf'], 'gamma': [2**i for i in range(-15,-4)], 'C': [2**i for i in range(-5,8)]},
		 					# {'kernel': ['linear'], 'C': [2**i for i in range(-8,9,2)]},
		 					# {'kernel': ['poly'], 'gamma': [2**i for i in range(-8,9,2)], 'C': [2**i for i in range(-8,9,2)], 'degree':[2,3,4]},
		 					]
		print "Start SVM CV ... "
		clf = GSCV(SVC(), tuned_parameters, cv=10)
		clf.fit(TrainX, TrainY)

		print("Best parameters set found on development set:")
		print(clf.best_params_)
		# print("Grid scores on development set:")
		# print()
		# for params, mean_score, scores in clf.grid_scores_:
		# 	print("%0.4f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))
		# print()

		print "Start SVM predicting ... "

		PredY = clf.predict(TestX)
		for i in range(len(TestY)):
			confuseMat[TestY[i]][PredY[i]] += 1

	return confuseMat




if __name__ == '__main__':
	print multi_SVM(needcv = True)
	# pass