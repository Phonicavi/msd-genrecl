# -*- coding:utf-8 -*-  

import h5py,os,math,pickle
import numpy as np 
from copy import deepcopy
import pickle,os,random
import threading
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import mixture
from sklearn.decomposition import PCA



import random

CONF_PHED = .6
DATA_DIV_RATIO = .85
H5_DATA_DIR = "../../fliter_h5_data/"


'''
TODO
Maybe use kmeans or GMM to cluster the timbres first instead of using the idx of maxValue
'''

class Song_PH:						## till now only implement for seg_timbre
	def __init__(self,fname,genre_,what = 'segments_timbre',preProcByCluster = True):
		try:
			f = h5py.File(fname,"r")
		except Exception,e:
			print Exception,":",e
		self.fname = fname
		self.genre = genre_
		
		self.PitchSeq = self.getPitchSeq(f,what,preProcByCluster)
		self.SeqNum = len(self.PitchSeq)


		f.close()

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


def preProcByClusterByKmeans(trains,tests):

	NUM_CLUSTER = 18

	newTrain = []
	newTest = []

	lenTrain = []
	lenTest = []

	ALL_FEA = []

	for singleGenre in trains:
		li = []
		for songs in singleGenre:
			ALL_FEA += songs
			li.append(len(songs))
		lenTrain.append(li)

	for singleGenre in tests:
		li = []
		for songs in singleGenre:
			ALL_FEA += songs
			li.append(len(songs))
		lenTest.append(li)

	print 'Start KMeans ... '
	# est = KMeans(n_clusters = NUM_CLUSTER)
	est = MiniBatchKMeans(n_clusters = NUM_CLUSTER,batch_size = int(len(ALL_FEA)/5),n_jobs=4)		## for speeding up
	est.fit(ALL_FEA)
	print 'Finish KMeans ... '
	lbls = est.labels_

	# print 'Start GMM ... '
	# est = mixture.GMM(n_components=NUM_CLUSTER, covariance_type='full',n_jobs=4)
	# est.fit(ALL_FEA)
	# print 'Finish GMM ... '
	# lbls = est.predict(ALL_FEA)

	# print 'Start PCA ...'
	# pca = PCA(n_components = 1,whiten = False,n_jobs=4)
	# lbls = pca.fit_transform(np.array(ALL_FEA))
	# print 'Finish PCA ... '

	cnt = 0

	for (i,singleGenre) in enumerate(trains):
		li = []
		for (j,songs) in enumerate(singleGenre):
			songseq = []
			for rp in range(lenTrain[i][j]):
				songseq.append(int(lbls[cnt]))
				cnt += 1
			li.append(songseq)
		newTrain.append(li)

	for (i,singleGenre) in enumerate(tests):
		li = []
		for (j,songs) in enumerate(singleGenre):
			songseq = []
			for rp in range(lenTest[i][j]):
				songseq.append(int(lbls[cnt]))
				cnt += 1
			li.append(songseq)
		newTest.append(li)

	assert(cnt == len(lbls)),"cnt != len(lbls)"


	return newTrain,newTest




def fetchData_PH(numNeeded,Genres,NeedReFetch,usedGenres = [1,1,1,1],fetch_what = 'segments_timbre',preProcByCluster = True):		## numNeeded like [250,250,250,250,250]
	assert(len(numNeeded) == len(Genres))
	assert(len(usedGenres) == len(Genres))
	# print Genres
	print "Start Fetching Data ..."
	if not NeedReFetch:
		try:
			with open('allGenreSongsTrain_PH.li','r') as f1,open('allGenreSongsTest_PH.li','r') as f2:
				allGenreSongsTrain = eval(f1.read())
				allGenreSongsTest = eval(f2.read())
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
				# if 1:
					tmpsongli.append(Song_PH(item,gen,what = fetch_what,preProcByCluster = True).PitchSeq)
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
			# 	tmpsongli.append(Song_PH(item,gen))
			# 	print item

		for t in TASK:
			t.start()
		for t in TASK:
			t.join();

		for i in range(len(Genres)):
			allGenreSongsTrain[i] = THD_RET[i][:int(n*DATA_DIV_RATIO)]
			allGenreSongsTest[i] = THD_RET[i][int(n*DATA_DIV_RATIO):]



		

		if not preProcByCluster:
			with open('allGenreSongsTrain_PH.li','w+') as f1 ,open('allGenreSongsTest_PH.li','w+') as f2:
				f1.write(str(allGenreSongsTrain));
				f2.write(str(allGenreSongsTest));

	

	if NeedReFetch and preProcByCluster:
		allGenreSongsTrain,allGenreSongsTest = preProcByClusterByKmeans(allGenreSongsTrain,allGenreSongsTest)
		with open('allGenreSongsTrain_PH.li','w+') as f1 ,open('allGenreSongsTest_PH.li','w+') as f2:
			f1.write(str(allGenreSongsTrain));
			f2.write(str(allGenreSongsTest));
	elif not preProcByCluster:
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
	fetchData_PH([200,200,200,200,200,200,200],['Jazz','Rap','Rock','Country','Blues','Latin','Electronic'],True,[1,1,1,1,1,1,1],fetch_what = 'segments_timbre',preProcByCluster = True);
		
