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
		clf = GSCV(SVC(), tuned_parameters, cv=5)
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



def NNet():

	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.advanced_activations import PReLU
	from keras.layers.normalization import BatchNormalization
	from keras.optimizers import SGD, Adadelta, Adagrad
	from keras.utils import np_utils
	from keras.regularizers import l2,l1
	from keras.layers.convolutional import Convolution2D, MaxPooling2D

	NeedReFetch = False
	allGenreSongsTrain,allGenreSongsTest = fetchData_eTA(NUM_NEED_PER_GENRE,GENRES,NeedReFetch)





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
	TrainY = np_utils.to_categorical(TrainY, len(GENRES))
	# print TrainY
	# TestY = np_utils.to_categorical(TestY, len(GENRES))


	model = Sequential()

	# model.add(Dense(output_dim=30, input_dim=len(TrainX[0]),W_regularizer=l1(0.005)))

	# model.add(Activation("tanh"))

	# model.add(Dense(output_dim=30))
	# # model.add(Activation('tanh'))
	# model.add(PReLU())
	# model.add(BatchNormalization())
	# model.add(Dropout(0.2))
	# # model.add(BatchNormalization())

	# # model.add(Dropout(0.2))
	
	# model.add(Dense(output_dim=30,W_regularizer=l2(0.005)))
	# model.add(Activation('relu'))
	# model.add(Dropout(0.2))

	# model.add(Dense(output_dim=30,W_regularizer=l2(0.002)))
	# model.add(Activation('tanh'))
	# model.add(Dropout(0.2))

	model.add(Dense(512,input_dim=len(TrainX[0])))
	
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	# model.add(Dropout(0.5))
	model.add(Dense(512,W_regularizer=l1(0.0005)))
	
	
	model.add(BatchNormalization())
	model.add(PReLU())
	# model.add(Dropout(0.5))
	for i in range (3):
		model.add(Dense(512,W_regularizer=l2(0.002)))
		model.add(BatchNormalization())
		model.add(PReLU())
		# model.add(Dropout(0.5))
		model.add(Dense(512,W_regularizer=l1(0.001)))
		model.add(BatchNormalization())
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))

	model.add(Dense(512,W_regularizer=l2(0.0002)))
	
	model.add(BatchNormalization())
	model.add(PReLU())
	# model.add(Dropout(0.2))

	model.add(Dense(512))
	
	
	model.add(BatchNormalization())
	model.add(Activation('tanh'))


	model.add(Dense(output_dim=len(GENRES)))
	model.add(Activation("softmax"))
	# sgd = SGD(lr=0.01,momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])

	model.fit(np.array(TrainX), TrainY, batch_size=int(len(TrainX)), nb_epoch=100,shuffle=True,verbose=1,validation_split=0.15)
	PredY = model.predict_classes(np.array(TestX), batch_size=int(len(TrainX)))

	confuseMat = [[0 for i in range(len(GENRES))] for j in range(len(GENRES))];



	for i in range(len(TestY)):
		# print PredY[i]
		confuseMat[TestY[i]][(PredY[i])] += 1
	return confuseMat







if __name__ == '__main__':
	# print multi_SVM(needcv = True)
	print NNet()
	# pass