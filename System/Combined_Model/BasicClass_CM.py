# -*- coding:utf-8 -*-  

import h5py,os,math,pickle
import numpy as np 
from copy import deepcopy
import pickle,os,random
import threading
from scipy.cluster.vq import whiten
from sklearn import preprocessing
from sklearn.decomposition import PCA,RandomizedPCA,KernelPCA,MiniBatchSparsePCA
from sklearn.externals import joblib
from sklearn.preprocessing import scale,StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.svm import SVC
import sys
sys.path.append('../')
sys.path.append('../HiddenMM/')
from FeatureSelection import featureSelection
from BasicClass_PH import preProcByClusterByKmeans

CONF_THED = .6
DATA_DIV_RATIO = .85
H5_DATA_DIR = "../../fliter_h5_data/"
# DATA_DIV_RATIO = .8
# H5_DATA_DIR = "../../fliter_h5_data_with_lyric/"



class Song_CM:
	def __init__(self,fname,genre_,
					sample_ratio = 0.2,window_size = 10,
					what = 'segments_timbre',preProcByCluster = True):
		try:
			f = h5py.File(fname,"r")
		except Exception,e:
			print Exception,":",e
		self.fname = fname
		self.genre = genre_
		
		self.allFeatures = self.getAllFeatures(f)

		self.SampleFeatures = self.getSampleFeatures(f,sample_ratio,window_size)
		self.SampleNum = len(self.SampleFeatures)



		self.PitchSeq = self.getPitchSeq(f,what,preProcByCluster)
		self.SeqNum = len(self.PitchSeq)


		f.close()
	def getE(self,vec):
		return float(sum(vec))/len(vec)

	def getAllFeatures(self,f):
		timbre_dim = 12
		timbreVec = list(f['analysis']['segments_timbre'])
		# print np.cov((np.array(f['analysis']['segments_timbre'][])).T)
		assert(len(timbreVec) > 0),'Not enough timbre'
		meanV = [(float(sum(item[i] for item in timbreVec))/len(timbreVec)) for i in range(len(timbreVec[0]))]
		covMat = [[0 for i in range(len(timbreVec[0]))] for j in range(len(timbreVec[0]))]
		for i in range(len(timbreVec[0])):
			tmpVec = []
			for item in timbreVec:
				tmpVec.append(item[i]-meanV[i])
			for j in range(len(timbreVec[0])):
				tmpVec2 = tmpVec[:]
				for k in range(len(timbreVec)):
					tmpVec2[k] = tmpVec[k] * (timbreVec[k][j] - meanV[j])
				covMat[i][j] = (self.getE(tmpVec2))

		covV = []
		for i in range(timbre_dim):
			for j in range(timbre_dim):
				if (j >= i):
					covV.append(covMat[i][j])

		##  till now 78+12=90 dim features

		beats_start = list(f['analysis']['beats_start'])
		assert(len(beats_start) > 0),'Not enough beats_start'
		beats_diff = [60.0/(beats_start[i+1]-beats_start[i]) for i in range(len(beats_start)-1)]

		mean_bpm = np.mean(np.asarray(beats_diff))
		cov_bpm = np.var(np.asarray(beats_diff))

		## till now 90 +1+1 = 92 dim features

		slmax = list(f['analysis']['segments_loudness_max']);
		assert(len(slmax)>0),'Not enough slmax'

		mean_slmax = np.mean(np.asarray(slmax))
		cov_slmax = np.var(np.asarray(slmax))

		## till now 92 +1+1 = 94 dim features

		sp = list(f['analysis']['segments_pitches'])
		assert(len(sp) > 0),'Not enough sp'


		dominNote = [0 for i in range(len(sp))]
		for i in range(len(sp)):
			(maxVal,dominNote[i]) = max((maxV,idx) for (idx,maxV) in enumerate(sp[i]))

		Trans = [0 for i in range(timbre_dim)]

		for i in range(len(sp)-1):
			trans = (dominNote[i+1]-dominNote[i])%timbre_dim
			Trans[trans] += 1
		Trans = [float(item)/(len(sp)-1) for item in Trans]

		## till now 94 +12 = 106 dim features

		allFeatures = np.hstack((meanV,covV,mean_bpm,cov_bpm,mean_slmax,cov_slmax,Trans))
		# allFeatures = np.hstack((meanV,mean_bpm,cov_bpm,mean_slmax,cov_slmax,Trans))


		# print len(allFeatures)

		# assert(len(allFeatures) == 106)

		return allFeatures




	def getPitchSeq(self,f,what,preProcByCluster):
		dim = 12
		pitches_arr =  np.array(f['analysis'][what])
		# print what
		assert(len(pitches_arr) > 1),"Not enough segs!"

		if preProcByCluster == True:
			return list(pitches_arr)

		pitches_seq = [0 for i in range(len(pitches_arr))]

		for i in range(len(pitches_arr)):
			(val,idx) = max((val,idx) for (idx,val) in enumerate(pitches_arr[i]))
			pitches_seq[i] = idx+1

		return pitches_seq


	def getSampleFeatures(self,f,sample_ratio,window_size):
		dim = 12
		timbres_arr =  np.array(f['analysis']['segments_timbre'])
		timbres_len = len(timbres_arr)
		assert(timbres_len > 2/sample_ratio and timbres_len > 2*window_size),"No enough timbre!"
		# print "here"

		sample_seg_start_idx = random.sample([i for i in range(timbres_len-window_size+1)],int((timbres_len-window_size+1)*sample_ratio))
		SampleFeatures = []
		# print timbres_arr
		for idx in sample_seg_start_idx:
			# print "here"
			tmp_arr = (timbres_arr[idx:idx+window_size]).T
			# print len(tmp_arr)
			assert(len(tmp_arr) == 12), "Not 12 dim!"
			fea = [np.mean(tmp_arr[i]) for i in range(len(tmp_arr))]
			# print len(fea)

			tmp_cov = [0.0 for i in range(dim)]

			for idx in range (dim):
				mean = fea[idx]
				tot = 0.0;
				for vecIndex in range(len(tmp_arr[idx])):
					error = (mean - tmp_arr[idx][vecIndex]) **2
					tot += error
				tot/=float(len(tmp_arr[idx]))
				tmp_cov[idx] = tot

			fea += tmp_cov
			assert(len(fea) == 24), "Not 24 length!"
			SampleFeatures.append(fea)

		# print np.cov((np.array(f['analysis']['segments_timbre'][])).T)
		return SampleFeatures



def generateData_CM(allGenreSongsTrain_part_eTA,
					allGenreSongsTest_part_eTA,
					allGenreSongsTrain_part_PH,
					allGenreSongsTest_part_PH,
					allGenreSongsTrain_part_VT,
					allGenreSongsTest_part_VT,
					):





	allGenreSongsTrain = [[] for i in range(len(allGenreSongsTrain_part_eTA))]
	allGenreSongsTest= [[] for i in range(len(allGenreSongsTest_part_eTA))]

	assert(len(allGenreSongsTrain_part_eTA) == len(allGenreSongsTest_part_VT))
	assert(len(allGenreSongsTrain_part_eTA[0]) == len(allGenreSongsTrain_part_PH[0]))

	NUM_GENRES = len(allGenreSongsTrain_part_eTA)

	## eTA part  	use SVM,LDA or GBDT
	##########################################################
	method_eTA = 'SVM'
	print '========================== method_eTA: ', method_eTA,' =========================='
	allGenreSongsTrain_part_eTA,allGenreSongsTest_part_eTA = featureSelection (allGenreSongsTrain_part_eTA,allGenreSongsTest_part_eTA,method = 'MIC',testmode = False,n_features_to_select = 90)


	TrainX = []
	TrainY = []
	TestX = []
	TestY = []
	for i in range(NUM_GENRES):
		for j in allGenreSongsTrain_part_eTA[i]:
			TrainX.append(j)
			TrainY.append(i)
		for k in allGenreSongsTest_part_eTA[i]:
			TestX.append(k)
			TestY.append(i)

	if method_eTA == 'SVM':
		tuned_parameters = [															## remained to be play with
							{'kernel': ['rbf'], 'gamma': [2**i for i in range(-15,-4)], 'C': [2**i for i in range(-5,8)]},
		 					]
		clf = GSCV(SVC(decision_function_shape='ovr'), tuned_parameters, cv=7)

	elif method_eTA == 'LDA':
		clf = LDA()
	else:
		clf = GradientBoostingClassifier(n_estimators=100, max_features = None,)
	clf.fit(TrainX, TrainY)

	for (idx,item) in enumerate(clf.decision_function(TrainX)):
		allGenreSongsTrain[TrainY[idx]].append(list(item))
	for (idx,item) in enumerate(clf.decision_function(TestX)):
		allGenreSongsTest[TestY[idx]].append(list(item))


	

	## PH part  	
	##########################################################
	method_PH = 'HMM'
	n_cpnts = 4
	covariance_type_ = 'tied'
	print '========================== method_PH: ', method_PH,' =========================='
	print "n_cpnts:",n_cpnts
	print 'covariance_type:',covariance_type_
	
	from hmmlearn import hmm

	hmmModel = [0 for i in range(NUM_GENRES)]





	def thd(i,seq_all,len_all, n_cpnts,n_it ,n_init):

		mdlList = []
		for ii in range(n_init):
			mdlList.append(hmm.GaussianHMM(n_components=n_cpnts,covariance_type=covariance_type_,n_iter = n_it).fit(np.array(seq_all), len_all))
		(maxv,idx) = max((maxScore,idx) for (idx,maxScore) in enumerate ([ mdl.score(seq_all,len_all) for mdl in mdlList]))
		hmmModel[i] = mdlList[idx]


	NeedTrain = False

	if NeedTrain:
		TASK = []
		for i in range(NUM_GENRES):
			seq_all = []
			len_all = []
			for (idx,trainSongs) in enumerate(allGenreSongsTrain_part_PH[i]):
				seq_all += [ [item] for item in trainSongs]
				# print trainSongs
				len_all.append(len(trainSongs))

			TASK.append(threading.Thread(target = thd,args = (i,seq_all,len_all,n_cpnts,150,4)))

		print "Starting Train model ..."
		for t in TASK:
			t.start()
		for t in TASK:
			t.join();
		joblib.dump(hmmModel,'hmmModel_CM.mdl',compress = 3)

	else:
		print "Starting Load model ..."
		hmmModel = joblib.load('hmmModel_CM.mdl')


	# TestY = []
	# PredY = []

	for i in range(NUM_GENRES):
		for (idx,trainSongs) in enumerate(allGenreSongsTrain_part_PH[i]):
			tsSeq = [ [item] for item in trainSongs ]
			allGenreSongsTrain[i][idx]+=list(scale([ mdl.score(tsSeq) for mdl in hmmModel]))

		for (idx,testSongs) in enumerate(allGenreSongsTest_part_PH[i]):
			tsSeq = [ [item] for item in testSongs ]
			allGenreSongsTest[i][idx]+=list(scale([ mdl.score(tsSeq) for mdl in hmmModel]))


	## VT part  	use RF or GBDT
	##########################################################

	method_VT = 'RF'
	print '========================== method_VT: ',method_VT,' =========================='

	TrainX = []
	TrainY = []
	for i in range(NUM_GENRES):
		for j in allGenreSongsTrain_part_VT[i]:
			for m in j:
				# print len(m)
				TrainX.append(m)
				TrainY.append(i)

	NeedTrain = False
	if NeedTrain:
		if method_VT == 'RF':
			clf = RandomForestClassifier( n_estimators=100,max_features = None,n_jobs = 4)
		else:
			clf = GradientBoostingClassifier(n_estimators=100, max_features = None)
		
		print "Starting Train model ..."
		clf.fit(TrainX,TrainY)
		joblib.dump(clf,'VT_CM.mdl',compress = 3)
	else:
		print "Starting Load model ..."
		clf = joblib.load('VT_CM.mdl')

	for i in range(NUM_GENRES):
		for (idx,trainSong) in enumerate(allGenreSongsTrain_part_VT[i]):
			trainSongY = clf.predict(trainSong)
			cnt = [0 for gen in range(NUM_GENRES)]
			for eachY in trainSongY:
				cnt[eachY] += 1
			allGenreSongsTrain[i][idx]+=list(scale(cnt))

			# (val,idx) = max((val,idx) for (idx,val) in enumerate(cnt))
			# confuseMat[i][idx] += 1
			# TestY.append(i)
			# PredY.append(idx)

		for (idx,testSong) in enumerate(allGenreSongsTest_part_VT[i]):
			testSongY = clf.predict(testSong)
			# print testSong
			cnt = [0 for gen in range(NUM_GENRES)]
			for eachY in testSongY:
				cnt[eachY] += 1
			allGenreSongsTest[i][idx]+=list(scale(cnt))






	return allGenreSongsTrain, allGenreSongsTest








	


''' ORDER: 0: eTA, 1: PH, 2: VT '''

def fetchData_CM(numNeeded,Genres,NeedReFetch,OnlyNeedReGenerate,usedGenres =[1,1,1,1],
					preProcByCluster = True

):		## numNeeded like [250,250,250,250,250]
	assert(len(numNeeded) == len(Genres))
	assert(len(usedGenres) == len(Genres))
	# assert((NeedReFetch and NeedReGenerate) or (not NeedReFetch)), 'Attention to NeedReGenerate and NeedReFetch!'
	print "Start Fetching Data ..."

	if not NeedReFetch:
		try:
			allGenreSongsTrain = joblib.load('allGenreSongsTrain_CM.pkl')
			allGenreSongsTest = joblib.load('allGenreSongsTest_CM.pkl')
			print len(allGenreSongsTrain)		
		except Exception,e:
			print Exception,':',e
	else:
		## TODO

		allGenreSongsTrain = [[] for i in range(len(Genres))]
		allGenreSongsTest= [[] for i in range(len(Genres))]

		# global THD_RET

		def thd(i,n,gen):
			# global THD_RET
			# print THD_RET

			cnt = 0;

			tmpfnameli = []
			tmpsongli = [[],[],[]]
			for fname in os.listdir(H5_DATA_DIR+gen):
				tmpfnameli.append(H5_DATA_DIR+gen+"/"+fname)
			assert(n<=len(tmpfnameli))
			random.shuffle(tmpfnameli)
			for item in tmpfnameli:
				try:
					tmpsongli[0].append(Song_CM(item,gen).allFeatures)
					tmpsongli[1].append(Song_CM(item,gen).PitchSeq)
					tmpsongli[2].append(Song_CM(item,gen).SampleFeatures)
					cnt += 1
					print item,cnt
					if cnt >= n:
						break 
				except Exception,e:
					print Exception,":",e
					print "@@@@ Discarded: " ,item
					continue
			assert(len(tmpsongli[0]) == n)

			THD_RET[i] = tmpsongli

		if not OnlyNeedReGenerate:

			THD_RET = [[] for i in range(len(Genres)) ]
			TASK = []

			for i in range(len(Genres)):
				n = numNeeded[i]
				gen = Genres[i]

				TASK.append(threading.Thread(target = thd,args = (i,n,gen)))

				tmpfnameli = []
				tmpsongli = [[],[],[]]
				
				# for fname in os.listdir(H5_DATA_DIR+gen):
				# 	tmpfnameli.append(H5_DATA_DIR+gen+"/"+fname)
				# assert(n<=len(tmpfnameli))
				# for item in random.sample(tmpfnameli,n):
				# 	tmpsongli.append(Song_TA(item,gen))
				# 	print item

			for t in TASK:
				t.start()
			for t in TASK:
				t.join();



			## eTA part
			##########################################################

			ALL_SONGS_FEATURES = []

			for i in range(len(Genres)):
				ALL_SONGS_FEATURES += THD_RET[i][0]

			# print len(ALL_SONGS_FEATURES[0]),len(ALL_SONGS_FEATURES)

			preProc = "scaling"

			if preProc == 'whiten':
				PREPORC_ALL_SONGS_FEATURES = list(whiten(np.array(ALL_SONGS_FEATURES)))
			elif preProc == 'scaling':
				PREPORC_ALL_SONGS_FEATURES = list(preprocessing.scale(np.array(ALL_SONGS_FEATURES)))
			else:
				PREPORC_ALL_SONGS_FEATURES = (ALL_SONGS_FEATURES)

			head = 0
			tail = 0

			allGenreSongsTrain_part_eTA = [[] for i in range(len(Genres))]
			allGenreSongsTest_part_eTA = [[] for i in range(len(Genres))]



			for i in range(len(Genres)):
				head = tail;
				tail += numNeeded[i]
				allGenreSongsTrain_part_eTA[i] = PREPORC_ALL_SONGS_FEATURES[head:tail] [:int(numNeeded[i]*DATA_DIV_RATIO)]
				allGenreSongsTest_part_eTA[i] = PREPORC_ALL_SONGS_FEATURES[head:tail][int(numNeeded[i]*DATA_DIV_RATIO):]
				assert(len(allGenreSongsTrain_part_eTA[i])+len(allGenreSongsTest_part_eTA[i]) == numNeeded[i])
			for i in range(len(usedGenres)):
				if (usedGenres[i] == 0):
					allGenreSongsTrain_part_eTA[i] = []
					allGenreSongsTest_part_eTA[i] = []
				for t in range(len(usedGenres) - sum(usedGenres)):
					allGenreSongsTrain_part_eTA.remove([])
					allGenreSongsTest_part_eTA.remove([])

			joblib.dump(allGenreSongsTrain_part_eTA,'allGenreSongsTrain_part_eTA_CM.pkl',compress = 3)
			joblib.dump(allGenreSongsTest_part_eTA,'allGenreSongsTest_part_eTA_CM.pkl',compress = 3)


			## PH part
			############################################################


			allGenreSongsTrain_part_PH = [[] for i in range(len(Genres))]
			allGenreSongsTest_part_PH = [[] for i in range(len(Genres))]

			for i in range(len(Genres)):
				allGenreSongsTrain_part_PH[i] = THD_RET[i][1][:int(numNeeded[i]*DATA_DIV_RATIO)]
				allGenreSongsTest_part_PH[i] = THD_RET[i][1][int(numNeeded[i]*DATA_DIV_RATIO):]

			for i in range(len(usedGenres)):
				if (usedGenres[i] == 0):
					allGenreSongsTrain_part_PH[i] = []
					allGenreSongsTest_part_PH[i] = []
			for t in range(len(usedGenres) - sum(usedGenres)):
				allGenreSongsTrain_part_PH.remove([])
				allGenreSongsTest_part_PH.remove([])	

			if preProcByCluster:	
				allGenreSongsTrain_part_PH,allGenreSongsTest_part_PH = preProcByClusterByKmeans(allGenreSongsTrain_part_PH,allGenreSongsTest_part_PH)

			joblib.dump(allGenreSongsTrain_part_PH,'allGenreSongsTrain_part_PH_CM.pkl',compress = 3)
			joblib.dump(allGenreSongsTest_part_PH,'allGenreSongsTest_part_PH_CM.pkl',compress = 3)



			## VT part
			############################################################
			allGenreSongsTrain_part_VT = [[] for i in range(len(Genres))]
			allGenreSongsTest_part_VT = [[] for i in range(len(Genres))]

			for i in range(len(Genres)):
				allGenreSongsTrain_part_VT[i] = THD_RET[i][2][:int(numNeeded[i]*DATA_DIV_RATIO)]
				allGenreSongsTest_part_VT[i] = THD_RET[i][2][int(numNeeded[i]*DATA_DIV_RATIO):]

			for i in range(len(usedGenres)):
				if (usedGenres[i] == 0):
					allGenreSongsTrain_part_VT[i] = []
					allGenreSongsTest_part_VT[i] = []
			for t in range(len(usedGenres) - sum(usedGenres)):
				allGenreSongsTrain_part_VT.remove([])
				allGenreSongsTest_part_VT.remove([])

			joblib.dump(allGenreSongsTrain_part_VT,'allGenreSongsTrain_part_VT_CM.pkl',compress = 3)
			joblib.dump(allGenreSongsTest_part_VT,'allGenreSongsTest_part_VT_CM.pkl',compress = 3)


		else:
			allGenreSongsTrain_part_eTA = joblib.load('allGenreSongsTrain_part_eTA_CM.pkl')
			allGenreSongsTest_part_eTA = joblib.load('allGenreSongsTest_part_eTA_CM.pkl')

			allGenreSongsTrain_part_PH = joblib.load('allGenreSongsTrain_part_PH_CM.pkl')
			allGenreSongsTest_part_PH = joblib.load('allGenreSongsTest_part_PH_CM.pkl')

			allGenreSongsTrain_part_VT = joblib.load('allGenreSongsTrain_part_VT_CM.pkl')
			allGenreSongsTest_part_VT = joblib.load('allGenreSongsTest_part_VT_CM.pkl')


		allGenreSongsTrain,allGenreSongsTest = generateData_CM(
									allGenreSongsTrain_part_eTA,
									allGenreSongsTest_part_eTA,
									allGenreSongsTrain_part_PH,
									allGenreSongsTest_part_PH,
									allGenreSongsTrain_part_VT,
									allGenreSongsTest_part_VT,
									)

		joblib.dump(allGenreSongsTrain,'allGenreSongsTrain_CM.pkl',compress = 3)
		joblib.dump(allGenreSongsTest,'allGenreSongsTest_CM.pkl',compress = 3)


	return allGenreSongsTrain,allGenreSongsTest




if __name__ == '__main__':
	fetchData_CM([10,10,10,10],['Jazz','Rap','Rock','Country'],True,False);
	# fetchData_CM(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,OnlyNeedReGenerate,USED_GENRES)













