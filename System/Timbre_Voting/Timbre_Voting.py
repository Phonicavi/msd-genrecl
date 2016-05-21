# -*- coding:utf-8 -*-  
from BasicClass_VT import *
import sys
import random
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.metrics import classification_report as clfr
from sklearn.svm import SVC
import math
from sklearn.externals import joblib

NUM_NEED_PER_GENRE = [200,200,200,200,200,200,200]
GENRES = ['Jazz','Rap','Rock','Country','Blues','Latin','Electronic']
USED_GENRES = [1,1,1,1,0,0,0]		## remained to be XJBplay



# NUM_NEED_PER_GENRE = [240,130,110,40,70]
# GENRES = ['Rock','Rap','Country','Electronic','Latin']
# USED_GENRES = [1,1,0,0,1]	



def multi_SVM_VT(needcv = False):
	NeedReFetch = False
	allGenreSongsTrain,allGenreSongsTest = fetchData_VT(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,USED_GENRES)

	# assert(len(allGenreSongsTrain[0][0]) == 106)

	TrainX = []
	TrainY = []
	TestX = []
	TestY = []
	for i in range(sum(USED_GENRES)):
		for j in allGenreSongsTrain[i]:
			for m in j:
				# print len(m)
				TrainX.append(m)
				TrainY.append(i)

	print len(TrainY)

	confuseMat = [[0 for i in range(sum(USED_GENRES))] for j in range(sum(USED_GENRES))];



	if not needcv:
		print "Start SVM training ... "

		needTrain = True

		## Warning: It will take long time to fit the model, 

		if needTrain:
			model = SVC()
			model.fit(TrainX,TrainY)
			joblib.dump(model,'multi_svm_vt.mdl',compress = 3)
		else:
			model = joblib.load('multi_svm_vt.mdl')
		# print model.score(TrainX,TrainY)

		for i in range(sum(USED_GENRES)):
			for testSong in allGenreSongsTest[i]:
				testSongY = model.predict(testSong)
				# print testSong
				cnt = [0 for gen in range(sum(USED_GENRES))]
				for eachY in testSongY:
					cnt[eachY] += 1
				print cnt
				(val,idx) = max((val,idx) for (idx,val) in enumerate(cnt))
				confuseMat[i][idx] += 1
				print i ,idx





		
	else:
		tuned_parameters = [															## remained to be play with
							{'kernel': ['rbf'], 'gamma': [2**i for i in range(-15,-4)], 'C': [2**i for i in range(-5,8)]},
		 					# {'kernel': ['linear'], 'C': [2**i for i in range(-8,9,2)]},
		 					# {'kernel': ['poly'], 'gamma': [2**i for i in range(-8,9,2)], 'C': [2**i for i in range(-8,9,2)], 'degree':[2,3,4]},
		 					]
		print "Start SVM CV ... "
		clf = GSCV(SVC(), tuned_parameters, cv=7)
		clf.fit(TrainX, TrainY)

		print("Best parameters set found on development set:")
		print(clf.best_params_)
		# print("Grid scores on development set:")
		# print()
		# for params, mean_score, scores in clf.grid_scores_:
		# 	print("%0.4f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))
		# print()

		print "Start SVM predicting ... "

		for i in range(sum(USED_GENRES)):
			for testSong in allGenreSongsTest[i]:
				testSongY = clf.predict(testSong)
				cnt = [0 for gen in range(sum(USED_GENRES))]
				for eachY in testSongY:
					cnt[eachY] += 1
				(val,idx) = max((val,idx) for (idx,val) in enumerate(cnt))
				confuseMat[i][idx] += 1

	return confuseMat


def OtherClassicalClassifier_VT():
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
	from sklearn.naive_bayes import GaussianNB
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
	from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

	NeedReFetch = False
	allGenreSongsTrain,allGenreSongsTest = fetchData_VT(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,USED_GENRES)


	classifiers = [
	# ("KNN",KNeighborsClassifier(n_neighbors = 1000)),
    # ("Decision Tree",DecisionTreeClassifier(criterion = 'gini')),
    ("Random Forest",RandomForestClassifier( n_estimators=100,max_features = None,n_jobs = 4)),
    # ("AdaBoost",AdaBoostClassifier( n_estimators=100,)),
    # ("Gaussian Naive Bayes",GaussianNB()),
    # ("LDA",LDA()),
    # ("QDA",QDA()),
    ("GBDT",GradientBoostingClassifier(n_estimators=100, max_features = None)),
    ]

	TrainX = []
	TrainY = []
	for i in range(sum(USED_GENRES)):
		for j in allGenreSongsTrain[i]:
			for m in j:
				# print len(m)
				TrainX.append(m)
				TrainY.append(i)

	for name, clf in classifiers:

		print "Start",name,"training ... "
		clf.fit(TrainX, TrainY)

		confuseMat = [[0 for i in range(sum(USED_GENRES))] for j in range(sum(USED_GENRES))];

		TestY = []
		PredY = []

		for i in range(sum(USED_GENRES)):
			for testSong in allGenreSongsTest[i]:
				testSongY = clf.predict(testSong)
				# print testSong
				cnt = [0 for gen in range(sum(USED_GENRES))]
				for eachY in testSongY:
					cnt[eachY] += 1
				(val,idx) = max((val,idx) for (idx,val) in enumerate(cnt))
				confuseMat[i][idx] += 1
				TestY.append(i)
				PredY.append(idx)



		print name,confuseMat
		print(clfr(TestY, PredY))

	return


if __name__ == '__main__':
	# print multi_SVM_VT(needcv = False)		## poisoned and training of this mdl will take a long long time
	OtherClassicalClassifier_VT()
