# -*- coding:utf-8 -*-  

import h5py,os,math,pickle
import numpy as np 
from copy import deepcopy
import pickle,os,random
import threading

CONF_THED = .6
DATA_DIV_RATIO = .85
H5_DATA_DIR = "../../fliter_h5_data/"



# DATA_DIV_RATIO = .8
# H5_DATA_DIR = "../../fliter_h5_data_with_lyric/"


class Song_TA:
	def __init__(self,fname,genre_):
		try:
			f = h5py.File(fname,"r")
		except Exception,e:
			print Exception,":",e
		self.fname = fname
		self.genre = genre_
		self.timbreVec, self.timbreVecConf = self.getTimbreVector(f)

		# if (len(self.timbreVec) == 0):
		# 	f.close()
		# 	assert(False),"No timbre features with "+fname


		# self.segPit = self.getSegPit(f)
		# self.segStart = self.getSegStart(f)
		# self.segLoudnessMax, self.segLoudnessStart  = self.getSegLoudness(f)
		self.mean_timbreVec = self.getMean_timbre()
		self.cov_timbreVec = self.getCov_timbre()

		self.timbreVec = []
		self.timbreVecConf = []

		f.close()


	def getTimbreVector(self,f):
		return list(f['analysis']['segments_timbre']),list(f['analysis']['segments_confidence'])

	def getSegStart(self,f):
		return list(f['analysis']['segments_start'])

	def getSegPit(self,f):
		return list(f['analysis']['segments_pitches'])

	def getSegLoudness(self,f):
		return list(f['analysis']['segments_loudness_max']),list(f['analysis']['segments_loudness_start'])

	def getMean_timbre(self):
		meanV = [(float(sum(item[i] for item in self.timbreVec))/len(self.timbreVec)) for i in range(len(self.timbreVec[0]))]
		return meanV

	def getE(self,vec):
		return float(sum(vec))/len(vec)

	def getCov_timbre(self):     ## 'full' not 'diag'
		covMat = [[0 for i in range(len(self.timbreVec[0]))] for j in range(len(self.timbreVec[0]))]
		for i in range(len(self.timbreVec[0])):
			# covMat.append([]);
			tmpVec = []
			for item in self.timbreVec:
				tmpVec.append(item[i]-self.mean_timbreVec[i])
			for j in range(len(self.timbreVec[0])):
				tmpVec2 = tmpVec[:]
				for k in range(len(self.timbreVec)):
					tmpVec2[k] = tmpVec[k] * (self.timbreVec[k][j] - self.mean_timbreVec[j])
				covMat[i][j] = (self.getE(tmpVec2))
		return covMat

class Centroid_Kmeans:
	def __init__ (self, fname):
		self.fname = fname
		self.dim = 12
		self.mean_timbreVec = self.initMean_timbre()
		self.cov_timbreVec = self.initCov_timbre()

	def initMean_timbre(self):
		return [0.0 for i in range(self.dim)]

	def initCov_timbre(self):
		ret = []
		for i in range(self.dim):
			ret.append([0.0 for j in range(self.dim)])
		return ret

	def addMean_timbre(self,vec):
		for i in range(len(vec)):
			self.mean_timbreVec[i] += vec[i]
	def addCov_timbre(self,cov):
		for i in range(self.dim):
			for j in range(self.dim):
				self.cov_timbreVec[i][j] += cov[i][j]

	def averMean_timbre(self,num):
		for i in range(len(self.mean_timbreVec)):
			self.mean_timbreVec[i] /= float(num)
	def averCov_timbre(self,num):
		for i in range(self.dim):
			for j in range(self.dim):
				self.cov_timbreVec[i][j] /= float(num)





def getKLdiv(meanV1,meanV2,cov1,cov2):
	dim = 12

	mat1 = np.asmatrix(cov1)
	mat2 = np.asmatrix(cov2)

	det1 = np.linalg.det(mat1)
	det2 = np.linalg.det(mat2)

	if (det1<=0 or det2<=0):
		# return 1000
		logcov_pq = 0;
		logcov_qp = 0;
	else:
		logcov_pq = math.log(det2/det1)
		logcov_qp = math.log(det1/det2)

	meanDiff_12 = np.asmatrix(np.asarray(meanV1)-np.asarray(meanV2))
	meanDiff_21 = np.asmatrix(np.asarray(meanV2)-np.asarray(meanV1))

	trcov_pq = np.trace(mat2.I*mat1)
	trcov_qp = np.trace(mat1.I*mat2)

	musq_pq = meanDiff_12*mat2.I*meanDiff_12.T
	musq_qp = meanDiff_21*mat1.I*meanDiff_21.T

	KL_pq = logcov_pq + trcov_pq + musq_pq - dim
	KL_qp = logcov_qp + trcov_qp + musq_qp - dim

	return 0.5*(KL_pq+KL_qp)






def fetchData_TA(numNeeded,Genres,NeedReFetch,usedGenres = [1,1,1,1]):		## numNeeded like [250,250,250,250,250]
	assert(len(numNeeded) == len(Genres))
	assert(len(usedGenres) == len(Genres))
	# print Genres
	print "Start Fetching Data ..."
	if not NeedReFetch:
		try:
			with open('allGenreSongsTrain_TA.pkl','rb') as f1,open('allGenreSongsTest_TA.pkl','rb') as f2:
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
			for item in random.sample(tmpfnameli,n):
				cnt += 1
				tmpsongli.append(Song_TA(item,gen))
				print item,cnt

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

		for i in range(len(Genres)):
			allGenreSongsTrain[i] = THD_RET[i][:int(numNeeded[i]*DATA_DIV_RATIO)]
			allGenreSongsTest[i] = THD_RET[i][int(numNeeded[i]*DATA_DIV_RATIO):]



		with open('allGenreSongsTrain_TA.pkl','wb') as f1 ,open('allGenreSongsTest_TA.pkl','wb') as f2:
			pickle.dump(allGenreSongsTrain,f1)
			pickle.dump(allGenreSongsTest,f2)

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
	fetchData_TA([20,20,20,20],['Jazz','Rap','Rock','Country'],True);












