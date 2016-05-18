import numpy as np 
from sklearn.decomposition import PCA,RandomizedPCA,KernelPCA,MiniBatchSparsePCA
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter 
from sklearn import preprocessing
from sklearn.externals import joblib
import random


CONF_PHED = .6
DATA_DIV_RATIO = .80

def fetchData_Lyric(numNeeded,Genres,NeedReFetch,usedGenres = [1,1,1,1],):		## numNeeded like [250,250,250,250,250]
	assert(len(numNeeded) == len(Genres))
	assert(len(usedGenres) == len(Genres))
	# print Genres
	print "Start Fetching Data ..."
	if not NeedReFetch:
		try:
			allGenreSongsTrain = joblib.load('allGenreSongsTrain_Lyric.pkl')
			allGenreSongsTest = joblib.load('allGenreSongsTest_Lyric.pkl')
		except Exception,e:
			print Exception,':',e
	else:
		## TODO
		allGenreSongsTrain = []
		allGenreSongsTest = []
		ALL_FEA = []
		LyricDic = (joblib.load('ori_BOW_data_for_each_song.pkl'))
		with open('my_tag_for_lyric.dic','r') as f:
			TagDic = eval(f.read())

		num_bf_pca = 1000
		num_af_pca = 8

		C = []
		CNT = []
		for (idx,gen) in enumerate(Genres):
			if (usedGenres[idx] == 0):
				continue
			all_feas  =[]
			cnt = 0
			for item in LyricDic.keys():
				if (TagDic[item] == gen):
					all_feas.append(LyricDic[item])
					ALL_FEA.append(LyricDic[item])
					cnt += 1

				if cnt >= numNeeded[idx]:
					break	

			CNT.append(cnt)
			c = np.array(all_feas).astype(bool).astype(int)
			C.append(sum(c)/float(cnt))

		C = np.array(C).T 
		varC = [(idx,item) for (idx,item) in enumerate(list(np.var(C,1)))]
		sort_varC = sorted(varC,key=itemgetter(1),reverse = True)[:num_bf_pca]
		select_word_idx_list = [idx for (idx,item) in sort_varC]

		for (i,fea) in enumerate(ALL_FEA):
			li = []
			for idx in range(len(fea)):
				if idx in select_word_idx_list:
					# print idx,len(fea)
					li.append(fea[idx])
			ALL_FEA[i] = li

		print 'Len of ALL_FEA: ',len(ALL_FEA)

		print 'Start PCA ... '

		pca = MiniBatchSparsePCA(n_components = num_af_pca,n_jobs = 4,verbose = 1,batch_size = len(ALL_FEA)/10)
		new_all_fea = pca.fit_transform(np.array(ALL_FEA))

		print '\nFinish PCA ... '

		allSongs = []
		head = 0
		tail = 0

		for gen in range(sum(usedGenres)):
			head = tail
			tail = tail + CNT[gen]
			allSongs.append(ALL_FEA[head:tail])
			random.shuffle(allSongs[gen])

		for i in range(sum(usedGenres)):
			allGenreSongsTrain.append(allSongs[i][:int(len(allSongs[i])*DATA_DIV_RATIO)])
			allGenreSongsTest.append(allSongs[i][int(len(allSongs[i])*DATA_DIV_RATIO):])

		joblib.dump(allGenreSongsTrain,'allGenreSongsTrain_Lyric.pkl',compress = 3)
		joblib.dump(allGenreSongsTest,'allGenreSongsTest_Lyric.pkl',compress = 3)


		
	return allGenreSongsTrain,allGenreSongsTest


if __name__ == '__main__':
	fetchData_Lyric([240,130,110,40,70],['Rock','Rap','Country','Electronic','Latin'],True,[1,1,1,0,0]);
		
