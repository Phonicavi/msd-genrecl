# -*- coding:utf-8 -*-  
from BasicClass_CM import *
import sys
import random
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.metrics import classification_report as clfr
from sklearn.svm import SVC
import math
import sys
sys.path.append('../')
from FeatureSelection import featureSelection



NUM_NEED_PER_GENRE = [200,200,200,200]
GENRES = ['Jazz','Rap','Rock','Country']
USED_GENRES = [1,1,1,1]					## remained to be XJBplay


# NUM_NEED_PER_GENRE = [240,130,110,40,70]
# GENRES = ['Rock','Rap','Country','Electronic','Latin']
# USED_GENRES = [1,1,0,0,1]	



'''
TODO:
	FeatureSelection()
'''



def multi_SVM(needcv = False):
	NeedReFetch = NEED_REFETCH
	OnlyNeedReGenerate = ONLY_NEED_REGENERATE
	allGenreSongsTrain,allGenreSongsTest = fetchData_CM(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,OnlyNeedReGenerate,USED_GENRES)
	# allGenreSongsTrain,allGenreSongsTest = featureSelection (allGenreSongsTrain,allGenreSongsTest,method = 'mean',testmode = True,n_features_to_select = 4)


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
		model = SVC(probability=True,decision_function_shape='ovo',kernel = 'rbf',gamma = 0.0078125, C = 8)
		model.fit(TrainX,TrainY)
		print "Start SVM predicting ... "
		PredY = model.predict(TestX)
		for i in range(len(TestY)):
			confuseMat[TestY[i]][PredY[i]] += 1
		print(clfr(TestY, PredY))
	else:
		tuned_parameters = [															## remained to be play with
							{'kernel': ['rbf'], 'gamma': [2**i for i in range(-25,-15)], 'C': [2**i for i in range(-15,-7)]},
		 					# {'kernel': ['linear'], 'C': [2**i for i in range(-8,9,2)]},
		 					# {'kernel': ['poly'], 'gamma': [2**i for i in range(-8,9,2)], 'C': [2**i for i in range(-8,9,2)], 'degree':[2,3,4]},
		 					]
		print "Start SVM CV ... "
		clf = GSCV(SVC(decision_function_shape='ovr'), tuned_parameters, cv=10)
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





def OtherClassicalClassifier():
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
	from sklearn.naive_bayes import GaussianNB
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
	from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

	NeedReFetch = NEED_REFETCH
	OnlyNeedReGenerate = ONLY_NEED_REGENERATE
	allGenreSongsTrain,allGenreSongsTest = fetchData_CM(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,OnlyNeedReGenerate,USED_GENRES)

	

	# allGenreSongsTrain,allGenreSongsTest = featureSelection (allGenreSongsTrain,allGenreSongsTest,method = 'MIC',testmode = False,n_features_to_select = 7)

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
    # ("Decision Tree",DecisionTreeClassifier()),
    ("Random Forest",RandomForestClassifier(criterion = 'gini', n_estimators=1000,max_features = 'auto')),
    # ("AdaBoost",AdaBoostClassifier( n_estimators=500,)),
    ("Gaussian Naive Bayes",GaussianNB()),
    ("LDA",LDA()),
    # ("QDA",QDA()),
    # ("GBDT",GradientBoostingClassifier(n_estimators=100, max_features = None)),
    ]


	for name, clf in classifiers:
		clf.fit(TrainX, TrainY)
		PredY = clf.predict(TestX)
		from sklearn.preprocessing import scale,StandardScaler,MinMaxScaler


		confuseMat = [[0 for i in range(sum(USED_GENRES))] for j in range(sum(USED_GENRES))];



		for i in range(len(TestY)):
			confuseMat[TestY[i]][(PredY[i])] += 1

		print name,confuseMat
		print(clfr(TestY, PredY))

	return
	# pass

def NNet():

	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.advanced_activations import PReLU
	from keras.layers.normalization import BatchNormalization
	from keras.optimizers import SGD, Adadelta, Adagrad,Adam,RMSprop
	from keras.utils import np_utils
	from keras.regularizers import l2,l1
	from keras.layers.convolutional import Convolution2D, MaxPooling2D

	NeedReFetch = NEED_REFETCH
	OnlyNeedReGenerate = ONLY_NEED_REGENERATE
	allGenreSongsTrain,allGenreSongsTest = fetchData_CM(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,OnlyNeedReGenerate ,USED_GENRES)
	# allGenreSongsTrain,allGenreSongsTest = featureSelection (allGenreSongsTrain,allGenreSongsTest,method = 'MIC',testmode = False,n_features_to_select = None)






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

	numNode = 20

	model.add(Dense(numNode,input_dim=len(TrainX[0]),W_regularizer=l1(0.1)))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))


	model.add(Dense(numNode,W_regularizer=l2(1),))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Dropout(0.2))



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

	model.fit(np.array(TrainX), TrainY, batch_size=int(len(TrainX)*.9),nb_epoch = 6000,shuffle=True,verbose=1,validation_split=0.2)
	PredY = model.predict_classes(np.array(TestX), batch_size=int(len(TrainX)*.9))

	confuseMat = [[0 for i in range(sum(USED_GENRES))] for j in range(sum(USED_GENRES))];

	print(clfr(TestY, PredY))

	for i in range(len(TestY)):
		confuseMat[TestY[i]][(PredY[i])] += 1
	return confuseMat





if __name__ == '__main__':
	NEED_REFETCH = True
	ONLY_NEED_REGENERATE = True
	# print multi_SVM(needcv = True)

	# NEED_REFETCH = False
	# ONLY_NEED_REGENERATE = True

	# print NNet()

	NEED_REFETCH = False
	ONLY_NEED_REGENERATE = True
	OtherClassicalClassifier()
	# pass