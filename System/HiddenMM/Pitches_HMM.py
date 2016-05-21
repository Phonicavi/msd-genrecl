# -*- coding:utf-8 -*-  
from BasicClass_PH import *
import sys
import random
from sklearn.metrics import classification_report as clfr
from hmmlearn import hmm
import math
from sklearn.externals import joblib

'''# ## Here is the diff, when you change the array, remember to set NeedFetch to True'''
NUM_NEED_PER_GENRE = [200,200,200,200,200,200,200]
GENRES = ['Jazz','Rap','Rock','Country','Blues','Latin','Electronic']


USED_GENRES = [1,0,1,1,0,0,0]					## remained to be XJBplay

# NUM_NEED_PER_GENRE = [240,130,110,40,70]
# GENRES = ['Rock','Rap','Country','Electronic','Latin']
# USED_GENRES = [1,1,0,0,1]	

def classifierByHmm(n_cpnts = 4,covariance_type_ = 'tied'):

	NeedReFetch = True


	'''
	Remember to change NeedReFetch when changing  used_what
	'''
	# used_what = 'segments_pitches'		## default option is to fetch 'segements_timbre'
	used_what = 'segments_timbre'		## default option is to fetch 'segements_timbre'

	allGenreSongsTrain,allGenreSongsTest = fetchData_PH(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,USED_GENRES,used_what,preProcByCluster = True)

	# global hmmModel
	hmmModel = [0 for i in range(sum(USED_GENRES))]


	print "n_cpnts:",n_cpnts
	print 'covariance_type:',covariance_type_
	print 'USED_GENRES: ',USED_GENRES
	print "used_what:",used_what

	def thd(i,seq_all,len_all, n_cpnts,n_it ,n_init):

		mdlList = []
		for ii in range(n_init):
			mdlList.append(hmm.GaussianHMM(n_components=n_cpnts,covariance_type=covariance_type_,n_iter = n_it).fit(np.array(seq_all), len_all))
		(maxv,idx) = max((maxScore,idx) for (idx,maxScore) in enumerate ([ mdl.score(seq_all,len_all) for mdl in mdlList]))
		hmmModel[i] = mdlList[idx]

	NeedTrain = True

	if NeedTrain:
		TASK = []
		for i in range(sum(USED_GENRES)):
			seq_all = []
			len_all = []
			for (idx,trainSongs) in enumerate(allGenreSongsTrain[i]):
				seq_all += [ [item] for item in trainSongs]
				# print trainSongs
				len_all.append(len(trainSongs))

			TASK.append(threading.Thread(target = thd,args = (i,seq_all,len_all,n_cpnts,150,4)))

		print "Starting Train model ..."

		for t in TASK:
			t.start()
		for t in TASK:
			t.join();

		# print hmmModel

		joblib.dump(hmmModel,'hmmModel.mdl',compress = 3)


	else:
		print "Starting Load model ..."
		hmmModel = joblib.load('hmmModel.mdl')


	confuseMat = [[0 for i in range(sum(USED_GENRES))] for j in range(sum(USED_GENRES))];

	TestY = []
	PredY = []

	for i in range(sum(USED_GENRES)):
		for (idx,testSongs) in enumerate(allGenreSongsTest[i]):
			tsSeq = [ [item] for item in testSongs ]
			(maxv,idx) = max((maxScore,idx) for (idx,maxScore) in enumerate ([ mdl.score(tsSeq) for mdl in hmmModel]))
			confuseMat[i][idx] += 1
			TestY.append(i)
			PredY.append(idx)


	print(clfr(TestY, PredY))
	return confuseMat


if __name__ == '__main__':
	print classifierByHmm()