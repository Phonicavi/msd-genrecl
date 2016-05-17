# -*- coding:utf-8 -*-  

import h5py,os,math,pickle
import numpy as np 
from copy import deepcopy
import pickle,os,random
import threading
from scipy.cluster.vq import whiten
from sklearn import preprocessing
import random

CONF_THED = .6
DATA_DIV_RATIO = .85
H5_DATA_DIR = "../fliter_h5_data/"

class Song_VT:						## till now only implement for seg_timbre
	def __init__(self,fname,genre_,sample_ratio = 0.2,window_size = 10):
		try:
			f = h5py.File(fname,"r")
		except Exception,e:
			print Exception,":",e
		self.fname = fname
		self.genre = genre_
		
		self.SampleFeatures = self.getSampleFeatures(f,sample_ratio,window_size)
		self.SampleNum = len(self.SampleFeatures)


		f.close()

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

def fetchData_VT(numNeeded,Genres,NeedReFetch,usedGenres =[1,1,1,1]):		## numNeeded like [250,250,250,250,250]
	assert(len(numNeeded) == len(Genres))
	assert(len(usedGenres) == len(Genres))
	print "Start Fetching Data ..."
	if not NeedReFetch:
		try:
			with open('allGenreSongsTrain_VT.li','r') as f1,open('allGenreSongsTest_VT.li','r') as f2:

				allGenreSongsTrain = eval(f1.read())
				allGenreSongsTest = eval(f2.read())


				##   too slow !!
				# allGenreSongsTrain = pickle.load(f1)
				# allGenreSongsTest = pickle.load(f2)
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
				# try:
				if 1:
					tmpsongli.append(Song_VT(item,gen).allFeatures)
					cnt += 1
					print item,cnt
					if cnt >= n:
						break 
				# except Exception,e:

					# print Exception,":",e
					# print "@@@@ Discarded: " ,item
					# continue
			assert(len(tmpsongli) == n)

			THD_RET[i] = tmpsongli

		# THD_RET = [[] for i in range(len(Genres)) ]
		# TASK = []

		# for i in range(len(Genres)):
		# 	n = numNeeded[i]
		# 	gen = Genres[i]

		# 	TASK.append(threading.Thread(target = thd,args = (i,n,gen)))

		# 	tmpfnameli = []
		# 	tmpsongli = []

		# 	# for fname in os.listdir(H5_DATA_DIR+gen):
		# 	# 	tmpfnameli.append(H5_DATA_DIR+gen+"/"+fname)
		# 	# assert(n<=len(tmpfnameli))
		# 	# for item in random.sample(tmpfnameli,n):
		# 	# 	tmpsongli.append(Song_TA(item,gen))
		# 	# 	print item

		# for t in TASK:
		# 	t.start()
		# for t in TASK:
		# 	t.join();

		# ALL_SONGS_FEATURES = []

		for i in range(len(Genres)):
			cnt  = 0
			tmpfnameli = []
			tmpsongli = []
			n = numNeeded[i]
			gen = Genres[i]
			for fname in os.listdir(H5_DATA_DIR+gen):
				tmpfnameli.append(H5_DATA_DIR+gen+"/"+fname)
			assert(n<=len(tmpfnameli))
			random.shuffle(tmpfnameli)
			for item in tmpfnameli:
				try:
				# if 1:
					tmpsongli.append(Song_VT(item,gen).SampleFeatures)
					cnt += 1
					print item,cnt
					if cnt >= n:
						break 
				except Exception,e:
					
					print Exception,":",e
					print "@@@@ Discarded: " ,item
					continue
			assert(len(tmpsongli) == n)

			# THD_RET[i] = tmpsongli
			allGenreSongsTrain[i] = tmpsongli[:int(n*DATA_DIV_RATIO)]
			allGenreSongsTest[i] = tmpsongli[int(n*DATA_DIV_RATIO):]

		''' preProcessing for this is a bit nasty -_-||  '''

		# preProc = "whiten"

		# if preProc == 'whiten':
		# 	PREPORC_ALL_SONGS_FEATURES = list(whiten(np.array(ALL_SONGS_FEATURES)))
		# elif preProc == 'scaling':
		# 	PREPORC_ALL_SONGS_FEATURES = list(preprocessing.scale(np.array(ALL_SONGS_FEATURES)))
		# else:
		# 	PREPORC_ALL_SONGS_FEATURES = (ALL_SONGS_FEATURES)
		# head = 0
		# tail = 0
		# for i in range(len(Genres)):
		# 	head = tail;
		# 	tail += numNeeded[i]
		# 	allGenreSongsTrain[i] = PREPORC_ALL_SONGS_FEATURES[head:tail] [:int(n*DATA_DIV_RATIO)]
		# 	allGenreSongsTest[i] = PREPORC_ALL_SONGS_FEATURES[head:tail][int(n*DATA_DIV_RATIO):]
		# 	assert(len(allGenreSongsTrain[i])+len(allGenreSongsTest[i]) == numNeeded[i])

		with open('allGenreSongsTrain_VT.li','w+') as f1 ,open('allGenreSongsTest_VT.li','w+') as f2:

			f1.write(str(allGenreSongsTrain));
			f2.write(str(allGenreSongsTest));

			# pickle.dump(allGenreSongsTrain,f1)
			# pickle.dump(allGenreSongsTest,f2)

	for i in range(len(usedGenres)):
		if (usedGenres[i] == 0):
			allGenreSongsTrain[i] = []
			allGenreSongsTest[i] = []
	for t in range(len(usedGenres) - sum(usedGenres)):
		allGenreSongsTrain.remove([])
		allGenreSongsTest.remove([])



	print "Done with Fetching Data ..."
	return allGenreSongsTrain,allGenreSongsTest




if __name__ == '__main__':
	fetchData_VT([200,200,200,200,200,200,200],['Jazz','Rap','Rock','Country','Blues','Latin','Electronic'],True,[1,1,1,1,1,1,1]);
