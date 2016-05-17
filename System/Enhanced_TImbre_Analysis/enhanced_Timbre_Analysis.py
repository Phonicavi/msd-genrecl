# -*- coding:utf-8 -*-  
from BasicClass_eTA import *
import sys
import random
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.metrics import classification_report as clfr
from sklearn.svm import SVC
import math

NUM_NEED_PER_GENRE = [200,200,200,200,200,200,200]
GENRES = ['Jazz','Rap','Rock','Country','Blues','Latin','Electronic']
USED_GENRES = [1,1,1,1,0,0,0]					## remained to be XJBplay



'''
TODO:
	FeatureSelection()
'''



def multi_SVM(needcv = False):
	NeedReFetch = False
	allGenreSongsTrain,allGenreSongsTest = fetchData_eTA(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,USED_GENRES)

	# assert(len(allGenreSongsTrain[0][0]) == 106)

	TrainX = []
	TrainY = []
	TestX = []
	TestY = []
	for i in range(sum(USED_GENRES)):
		for j in allGenreSongsTrain[i]:
			TrainX.append(j)
			TrainY.append(i)
		for k in allGenreSongsTest[i]:
			TestX.append(k)
			TestY.append(i)
	confuseMat = [[0 for i in range(sum(USED_GENRES))] for j in range(sum(USED_GENRES))];
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

		PredY = clf.predict(TestX)

		print(clfr(TestY, PredY))

		for i in range(len(TestY)):
			confuseMat[TestY[i]][PredY[i]] += 1

	return confuseMat



def NNet():

	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.advanced_activations import PReLU
	from keras.layers.normalization import BatchNormalization
	from keras.optimizers import SGD, Adadelta, Adagrad,Adam,RMSprop
	from keras.utils import np_utils
	from keras.regularizers import l2,l1
	from keras.layers.convolutional import Convolution2D, MaxPooling2D

	NeedReFetch = False
	allGenreSongsTrain,allGenreSongsTest = fetchData_eTA(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,USED_GENRES)





	TrainX = []
	TrainY = []
	TestX = []
	TestY = []
	for i in range(sum(USED_GENRES)):
		for j in allGenreSongsTrain[i]:
			TrainX.append(j)
			TrainY.append(i)
		for k in allGenreSongsTest[i]:
			TestX.append(k)
			TestY.append(i)
	TrainY = np_utils.to_categorical(TrainY, sum(USED_GENRES))


	## the network remained to be XJBplay

	model = Sequential()

	numNode = 10

	model.add(Dense(numNode,input_dim=len(TrainX[0]),W_regularizer=l1(0.1)))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Dropout(0.2))


	model.add(Dense(numNode,W_regularizer=l2(1),))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Dropout(0.3))



	# model.add(Dense(numNode,W_regularizer=l1(0.1)))
	# model.add(BatchNormalization())
	# model.add(PReLU())
	# model.add(Dropout(0.5))



		



	model.add(Dense(output_dim=sum(USED_GENRES)))
	model.add(Activation("softmax"))
	model.compile(loss='categorical_crossentropy'
				,optimizer="adam"
				,metrics=["accuracy"]
				)

	model.fit(np.array(TrainX), TrainY, batch_size=int(len(TrainX)*.9),nb_epoch = 6000,shuffle=True,verbose=1,validation_split=0.15)
	PredY = model.predict_classes(np.array(TestX), batch_size=int(len(TrainX)*.9))

	confuseMat = [[0 for i in range(sum(USED_GENRES))] for j in range(sum(USED_GENRES))];

	print(clfr(TestY, PredY))

	for i in range(len(TestY)):
		confuseMat[TestY[i]][(PredY[i])] += 1
	return confuseMat


def OtherClassicalClassifier():
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
	from sklearn.naive_bayes import GaussianNB
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
	from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

	NeedReFetch = False
	allGenreSongsTrain,allGenreSongsTest = fetchData_eTA(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,USED_GENRES)

	# assert(len(allGenreSongsTrain[0][0]) == 106)

	TrainX = []
	TrainY = []
	TestX = []
	TestY = []
	for i in range(sum(USED_GENRES)):
		for j in allGenreSongsTrain[i]:
			TrainX.append(j)
			TrainY.append(i)
		for k in allGenreSongsTest[i]:
			TestX.append(k)
			TestY.append(i)

	## remained to be XJBplay

	classifiers = [
    ("Decision Tree",DecisionTreeClassifier()),
    ("Random Forest",RandomForestClassifier( n_estimators=50,max_features = None)),
    ("AdaBoost",AdaBoostClassifier( n_estimators=50,)),
    ("Gaussian Naive Bayes",GaussianNB()),
    ("LDA",LDA()),
    ("QDA",QDA()),
    ("GBDT",GradientBoostingClassifier(n_estimators=100, max_features = None)),
    ]

	for name, clf in classifiers:
		clf.fit(TrainX, TrainY)
		PredY = clf.predict(TestX)

		confuseMat = [[0 for i in range(sum(USED_GENRES))] for j in range(sum(USED_GENRES))];



		for i in range(len(TestY)):
			confuseMat[TestY[i]][(PredY[i])] += 1

		print name,confuseMat
		print(clfr(TestY, PredY))

	return
	# pass





if __name__ == '__main__':
	# print multi_SVM(needcv = True)
	print NNet() 
	# OtherClassicalClassifier()
	# pass