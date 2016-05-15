# -*- coding:utf-8 -*-  

import h5py,os,math,pickle
import numpy as np 
from copy import deepcopy
import pickle,os,random
import threading
from scipy.cluster.vq import whiten

CONF_THED = .6
DATA_DIV_RATIO = .85
H5_DATA_DIR = "../fliter_h5_data/"


class Song_eTA:
	def __init__(self,fname,genre_):
		try:
			f = h5py.File(fname,"r")
		except Exception,e:
			print Exception,":",e
		self.fname = fname
		self.genre = genre_
		
		self.allFeatures = self.getAllFeatures(f)


		f.close()
	def getE(self,vec):
		return float(sum(vec))/len(vec)

	def getAllFeatures(self,f):
		timbre_dim = 12
		timbreVec = list(f['analysis']['segments_timbre'])
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

		# print len(allFeatures)

		# assert(len(allFeatures) == 106)

		return allFeatures




def fetchData_eTA(numNeeded,Genres,NeedReFetch):		## numNeeded like [250,250,250,250,250]
	assert(len(numNeeded) == len(Genres))
	print "Start Fetching Data ..."
	if not NeedReFetch:
		try:
			with open('allGenreSongsTrain_eTA.pkl','rb') as f1,open('allGenreSongsTest_eTA.pkl','rb') as f2:
				allGenreSongsTrain = pickle.load(f1)
				allGenreSongsTest = pickle.load(f2)
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
			tmpsongli = []
			for fname in os.listdir(H5_DATA_DIR+gen):
				tmpfnameli.append(H5_DATA_DIR+gen+"/"+fname)
			assert(n<=len(tmpfnameli))
			random.shuffle(tmpfnameli)
			for item in tmpfnameli:
				try:
					tmpsongli.append(Song_eTA(item,gen).allFeatures)
					cnt += 1
					print item,cnt
					if cnt >= n:
						break 
				except Exception,e:
					print Exception,":",e
					print "@@@@ Discarded: " ,item
					continue
			assert(len(tmpsongli) == n)

			THD_RET[i] = tmpsongli

		THD_RET = [[] for i in range(len(Genres)) ]
		TASK = []

		for i in range(len(Genres)):
			n = numNeeded[i]
			gen = Genres[i]

			TASK.append(threading.Thread(target = thd,args = (i,n,gen)))

			tmpfnameli = []
			tmpsongli = []

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

		ALL_SONGS_FEATURES = []

		for i in range(len(Genres)):
			ALL_SONGS_FEATURES += THD_RET[i]

		# print len(ALL_SONGS_FEATURES[0]),len(ALL_SONGS_FEATURES)
		WHITEN_ALL_SONGS_FEATURES = list(whiten(np.array(ALL_SONGS_FEATURES)))
		# WHITEN_ALL_SONGS_FEATURES = (ALL_SONGS_FEATURES)


		head = 0
		tail = 0
		for i in range(len(Genres)):
			head = tail;
			tail += numNeeded[i]
			allGenreSongsTrain[i] = WHITEN_ALL_SONGS_FEATURES[head:tail] [:int(n*DATA_DIV_RATIO)]
			allGenreSongsTest[i] = WHITEN_ALL_SONGS_FEATURES[head:tail][int(n*DATA_DIV_RATIO):]
			assert(len(allGenreSongsTrain[i])+len(allGenreSongsTest[i]) == numNeeded[i])



		with open('allGenreSongsTrain_eTA.pkl','wb') as f1 ,open('allGenreSongsTest_eTA.pkl','wb') as f2:
			pickle.dump(allGenreSongsTrain,f1)
			pickle.dump(allGenreSongsTest,f2)
	print "Done with Fetching Data ..."
	return allGenreSongsTrain,allGenreSongsTest




if __name__ == '__main__':
	fetchData_eTA([200,200,200,200],['Jazz','Rap','Rock','Country'],True);












