# -*- coding:utf-8 -*-  

from BasicClass_TA import *
import sys
import random
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report as clfr
from sklearn import mixture
import sys


NUM_NEED_PER_GENRE = [200,200,200,200,200,200,200]
GENRES = ['Jazz','Rap','Rock','Country','Blues','Latin','Electronic']
USED_GENRES = [1,1,1,1,0,0,0]		## remained to be XJBplay

# NUM_NEED_PER_GENRE = [240,130,110,40,70]
# GENRES = ['Rock','Rap','Country','Electronic','Latin']
# USED_GENRES = [1,1,0,0,1]

def getF(select_Idx,left_Idx,SingleGenreSongs):
	F = [0.0 for i in range(len(left_Idx))]
	for i in range(len(left_Idx)):
		tmpF_for_i = [ getKLdiv(SingleGenreSongs[left_Idx[i]].mean_timbreVec,SingleGenreSongs[select_Idx[j]].mean_timbreVec,
						SingleGenreSongs[left_Idx[i]].cov_timbreVec,SingleGenreSongs[select_Idx[j]].cov_timbreVec ) for j in range(len(select_Idx))]
		(minv,idx) = min((minv,idx) for (idx,minv) in enumerate(tmpF_for_i))
		# print tmpF_for_i
		F[i] = minv
	# print F
	F = list(np.array(F)/sum(F))
	return F


def Inner_Kmeans(SingleGenreSongs,numOfClusters):

	## using self-defined kmeans for inner kmeans

	## first kmeans++

	# print 'Start Kmeans++ for Inner_Kmeans'
	# First_Random_Choice_Idx = random.randint(0,len(SingleGenreSongs)-1);
	# left = len(SingleGenreSongs)-1;
	# left_Idx = [i for i in range(len(SingleGenreSongs))]
	# left_Idx.remove(First_Random_Choice_Idx)
	# select_Idx = [First_Random_Choice_Idx]
	# assert(len(left_Idx) == left)




	# while not numOfClusters == len(select_Idx) :
	# 	F = getF(select_Idx,left_Idx,SingleGenreSongs)
	# 	select_cluster = 0;
	# 	a = random.random();
	# 	while True:
	# 		a -= F[select_cluster]
	# 		if (a<0):
	# 			break
	# 		select_cluster += 1
	# 	nc = left_Idx[select_cluster]
	# 	select_Idx.append(nc)
	# 	left_Idx.remove(nc)
	# print 'Inner_Kmeans initial_cluster: ',select_Idx




	# cluster_centroids = []	
	# for (i,item) in enumerate(select_Idx):
	# 	cluster_centroids.append(Centroid_Kmeans(str(item)))
	# 	cluster_centroids[i].addMean_timbre(SingleGenreSongs[item].mean_timbreVec)
	# 	cluster_centroids[i].addCov_timbre(SingleGenreSongs[item].cov_timbreVec)
	# ori_cluster_centroids = deepcopy(cluster_centroids)



	iter_num = 100
	init_num = 5

	min_error = sys.maxint
	best_cluster_centroids = []
	best_lbls = []
	pre_lbls = [] 

	# print 'fsfdfd'


	for init in range(init_num):
		this_time_lbls = [-1 for i in range(len(SingleGenreSongs))]
		pre_lbls = [] 



		# print 'Start Kmeans++ for Inner_Kmeans'
		First_Random_Choice_Idx = random.randint(0,len(SingleGenreSongs)-1);
		left = len(SingleGenreSongs)-1;
		left_Idx = [i for i in range(len(SingleGenreSongs))]
		left_Idx.remove(First_Random_Choice_Idx)
		select_Idx = [First_Random_Choice_Idx]
		assert(len(left_Idx) == left)




		while not numOfClusters == len(select_Idx) :
			F = getF(select_Idx,left_Idx,SingleGenreSongs)
			select_cluster = 0;
			a = random.random();
			while True:
				a -= F[select_cluster]
				if (a<0):
					break
				select_cluster += 1
			nc = left_Idx[select_cluster]
			select_Idx.append(nc)
			left_Idx.remove(nc)
		# print 'Inner_Kmeans initial_cluster: ',select_Idx




		cluster_centroids = []	
		for (i,item) in enumerate(select_Idx):
			cluster_centroids.append(Centroid_Kmeans(str(item)))
			cluster_centroids[i].addMean_timbre(SingleGenreSongs[item].mean_timbreVec)
			cluster_centroids[i].addCov_timbre(SingleGenreSongs[item].cov_timbreVec)
		# ori_cluster_centroids = deepcopy(cluster_centroids)




		# print 'init = ',init,'maxIter = ',iter_num
		for it in range(iter_num):
			sys.stdout.write('>')
			sys.stdout.flush()

			error = 0.0
			num_in_centroid = [0 for i in range(len(cluster_centroids))]
			new_cluster_centroids = []
			for i in range (len(cluster_centroids)):
				new_cluster_centroids.append(Centroid_Kmeans(str(i)))
			for (songidx,song) in enumerate (SingleGenreSongs):
				KL_dist = []
				for centroid in (cluster_centroids):
					KL_dist.append(getKLdiv(song.mean_timbreVec,centroid.mean_timbreVec,song.cov_timbreVec,centroid.cov_timbreVec))	# pay attention to something need to be discarded
				val,idx = min((val,idx) for (idx,val) in enumerate(KL_dist))
				error += val
				this_time_lbls[songidx] = idx
				num_in_centroid[idx] += 1
				new_cluster_centroids[idx].addMean_timbre(song.mean_timbreVec)
				new_cluster_centroids[idx].addCov_timbre(song.cov_timbreVec)


			for i in range(len(new_cluster_centroids)):
				if (num_in_centroid[i] > 0):
					# print "# in cluster ",i,": ",num_in_centroid[i]
					new_cluster_centroids[i].averMean_timbre(num_in_centroid[i])
					new_cluster_centroids[i].averCov_timbre(num_in_centroid[i])
			# print this_time_lbls
			cluster_centroids = deepcopy(new_cluster_centroids)
			if (pre_lbls == this_time_lbls):
				# print this_time_lbls
				break
			else:
				pre_lbls = deepcopy(this_time_lbls)
		# print '\niternum = ',it

		if (error < min_error):
			error = min_error
			best_cluster_centroids = deepcopy(cluster_centroids)
			best_lbls = deepcopy(this_time_lbls)

	# print len(best_cluster_centroids)
	# print 

	assert(len(best_cluster_centroids) == numOfClusters ),'len(best_cluster_centroids) != numOfClusters'

	lbls = best_lbls
	# print lbls
	# print '================================'



	## using sklearn kmeans for inner kmeans
	# means = [item.mean_timbreVec for item in SingleGenreSongs]
	# means_array = np.array(means)
	# model = KMeans(n_clusters = numOfClusters,n_init = 10)
	# model.fit(means)
	# lbls = model.labels_

	clusters_mean = [[] for i in range(numOfClusters)];
	clusters_cov = [[] for i in range(numOfClusters)];
	clusters_cnt = [0 for i in range(numOfClusters)];
	ret_clusters_mean = [[] for i in range(numOfClusters)];
	ret_clusters_cov = [[] for i in range(numOfClusters)];

	assert(len(lbls) == len(SingleGenreSongs))

	for i in range(len(lbls)):
		lb = lbls[i]
		clusters_mean[lb].append(np.array(SingleGenreSongs[i].mean_timbreVec))
		clusters_cov[lb].append(np.array(SingleGenreSongs[i].cov_timbreVec))
		clusters_cnt[lb] += 1


	for i in range(numOfClusters):
		# print type(clusters_mean[i][0])
		ret_clusters_mean[i] = sum(clusters_mean[i])
		ret_clusters_cov[i] = sum(clusters_cov[i])
		# print type(ret_clusters_mean[i][0])
		ret_clusters_mean[i] = [(item)/clusters_cnt[i] for item in ret_clusters_mean[i]]
		ret_clusters_cov[i] = [(item)/clusters_cnt[i] for item in ret_clusters_cov[i]]

	return ret_clusters_mean,ret_clusters_cov




def Inner_GMM(SingleGenreSongs,numOfClusters):
	means = [item.mean_timbreVec for item in SingleGenreSongs]
	means_array = np.array(means)
	model = mixture.GMM(n_components = numOfClusters,covariance_type = 'full',n_init = 10)
	model.fit(means)

	lbls = model.predict(means)

	clusters_mean = [[] for i in range(numOfClusters)];
	clusters_cov = [[] for i in range(numOfClusters)];
	clusters_cnt = [0 for i in range(numOfClusters)];
	ret_clusters_mean = [[] for i in range(numOfClusters)];
	ret_clusters_cov = [[] for i in range(numOfClusters)];

	assert(len(lbls) == len(SingleGenreSongs))

	for i in range(len(lbls)):
		lb = lbls[i]
		clusters_mean[lb].append(np.array(SingleGenreSongs[i].mean_timbreVec))
		clusters_cov[lb].append(np.array(SingleGenreSongs[i].cov_timbreVec))
		clusters_cnt[lb] += 1


	for i in range(numOfClusters):
		# print type(clusters_mean[i][0])
		ret_clusters_mean[i] = sum(clusters_mean[i])
		ret_clusters_cov[i] = sum(clusters_cov[i])
		# print type(ret_clusters_mean[i][0])
		ret_clusters_mean[i] = [(item)/clusters_cnt[i] for item in ret_clusters_mean[i]]
		ret_clusters_cov[i] = [(item)/clusters_cnt[i] for item in ret_clusters_cov[i]]
	return ret_clusters_mean,ret_clusters_cov

	# return model.means_,model.covars_








def Kmeans_KL_train(allGenreSongs,numofGenres,assign_cluster = [[0],[1],[2],[3]],inner = "Kmeans"):				## type(allGenreSongs) = [li,li,li,...,li]
	iter_num = 30
	assert(numofGenres == len(assign_cluster))
	allSongs = []
	for i in range (numofGenres):
		allSongs += allGenreSongs[i]


	## Random select cluster centroid   >_<  seems to be not good
	# cluster_centroids = []
	# for i in range (numofGenres):
	# 	for j in assign_cluster[i]:
	# 		cluster_centroids.append(Centroid_Kmeans(str(i)))
	# 		sel = int(random.random()*len(allGenreSongs[i]))%len(allGenreSongs[i])
	# 		cluster_centroids[j].addMean_timbre(allGenreSongs[i][sel].mean_timbreVec)
	# 		cluster_centroids[j].addCov_timbre(allGenreSongs[i][sel].cov_timbreVec)


	## Using inner_kmeans select cluster centroid  
	cluster_centroids = []
	for i in range (numofGenres):
		if inner == 'Kmeans':
			c_means,c_covs = Inner_Kmeans(allGenreSongs[i],len(assign_cluster[i]));
		elif inner == "GMM":
			c_means,c_covs = Inner_GMM(allGenreSongs[i],len(assign_cluster[i]));
		else:
			assert(False),"Not else inner clustering supported!"
		# print c_means
		assert(len(c_means) == len(assign_cluster[i]))

		for idx in range(len(assign_cluster[i])):
			j = assign_cluster[i][idx]
			cluster_centroids.append(Centroid_Kmeans(str(i)))

			cluster_centroids[j].addMean_timbre(c_means[idx])
			cluster_centroids[j].addCov_timbre(c_covs[idx])
	print "\nlen of cluster_centroids", len(cluster_centroids)

	this_time_lbls = [-1 for i in range(len(allSongs))]
	pre_lbls = [] 

	for it in range(iter_num):
		print "#iter",it,"..."
		error = 0.0
		num_in_centroid = [0 for i in range(len(cluster_centroids))]
		new_cluster_centroids = []
		for i in range (len(cluster_centroids)):
			new_cluster_centroids.append(Centroid_Kmeans(str(i)))
		for (songidx,song) in enumerate(allSongs):
			KL_dist = []
			for centroid in cluster_centroids:
				KL_dist.append(getKLdiv(song.mean_timbreVec,centroid.mean_timbreVec,song.cov_timbreVec,centroid.cov_timbreVec))	# pay attention to something need to be discarded
			val,idx = min((val,idx) for (idx,val) in enumerate(KL_dist))
			this_time_lbls[songidx] = idx
			error += val
			num_in_centroid[idx] += 1
			new_cluster_centroids[idx].addMean_timbre(song.mean_timbreVec)
			new_cluster_centroids[idx].addCov_timbre(song.cov_timbreVec)

		for i in range(len(new_cluster_centroids)):
			if (num_in_centroid[i] > 0):
				# print "# in cluster ",i,": ",num_in_centroid[i]
				new_cluster_centroids[i].averMean_timbre(num_in_centroid[i])
				new_cluster_centroids[i].averCov_timbre(num_in_centroid[i])
		cluster_centroids = deepcopy(new_cluster_centroids)
		if (pre_lbls == this_time_lbls):
		# print this_time_lbls
			break
		else:
			pre_lbls = deepcopy(this_time_lbls)

	return error,cluster_centroids


def Kmeans_KL(cluster_num_of_each_genre,inner = "Kmeans",n_init = 3):
	## FetchData
	NeedReFetch = False
	assert(len(cluster_num_of_each_genre) == sum(USED_GENRES))
	allGenreSongsTrain,allGenreSongsTest = fetchData_TA(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,USED_GENRES)

	print "Start Kmeans training ..."

	assign_cluster = [[] for i in range(sum(USED_GENRES))]

	idx = 0;
	for gen in range(sum(USED_GENRES)):
		for n in range(cluster_num_of_each_genre[gen]):
			assign_cluster[gen].append(idx)
			idx += 1


	print assign_cluster

	(error,model)  = min((error,model) for (error,model) in ( [Kmeans_KL_train(allGenreSongsTrain,sum(USED_GENRES),assign_cluster,inner) for init in range(n_init)]));
	print "Finish Kmeans training ..."

	confuseMat = [[0 for i in range(sum(USED_GENRES))] for j in range(sum(USED_GENRES))];

	print "Start Kmeans predicting ..."
	cnt = 0;
	
	TestY = []
	PredY = []

	for i in range(sum(USED_GENRES)):
		for song in allGenreSongsTest[i]:
			cnt += 1
			print "Dealing with testSongs ", cnt
			KL_dist = []
			for centroid in model:
				KL_dist.append(getKLdiv(song.mean_timbreVec,centroid.mean_timbreVec,song.cov_timbreVec,centroid.cov_timbreVec))
			val,idx = min((val,idx) for (idx,val) in enumerate(KL_dist))
			gen = -1
			for j in range(len(assign_cluster)):
				if idx in assign_cluster[j]:
					gen = j
					break
			confuseMat[i][gen] += 1
			TestY.append(i)
			PredY.append(gen)

	print(clfr(TestY, PredY))
	print "Finish Kmeans predicting ..."
	return confuseMat

def Knn_KL(K_value):
	## FetchData
	NeedReFetch = False
	allGenreSongsTrain,allGenreSongsTest = fetchData_TA(NUM_NEED_PER_GENRE,GENRES,NeedReFetch,USED_GENRES)

	confuseMat = [[0 for i in range(sum(USED_GENRES))] for j in range(sum(USED_GENRES))];

	print "Start Knn ..."

	cnt = 0;

	TestY = []
	PredY = []
	for i in range(sum(USED_GENRES)):
		for testSong in allGenreSongsTest[i]:
			cnt += 1
			print "Dealing with testSongs ", cnt
			k_nearest = [[sys.maxint,-1] for j in range(K_value)]		## (KL_dist,cluster_centroid)

			for j in range(sum(USED_GENRES)):
				for trainSong in allGenreSongsTrain[j]:
					tp,idx = max((tp,idx) for (idx,tp) in enumerate(k_nearest))
					cen_idx = k_nearest[idx][1]
					tmp_dis = getKLdiv(testSong.mean_timbreVec,trainSong.mean_timbreVec,testSong.cov_timbreVec,trainSong.cov_timbreVec)

					if  tmp_dis < k_nearest[idx][0]:
						k_nearest[idx][0] = tmp_dis
						k_nearest[idx][1] = j
			voting = [0 for j in range(sum(USED_GENRES))]
			sum_of_dist = [0.0 for j in range(sum(USED_GENRES))]

			for j in range(K_value):
				voting[k_nearest[j][1]]+=1
				sum_of_dist[k_nearest[j][1]] += k_nearest[j][0]

			max_cnt = -1;
			min_dist = sys.maxint
			max_idx = -1;

			for j in range(sum(USED_GENRES)):
				if (voting[j] > max_cnt) or (voting[j] == max_cnt and min_dist > sum_of_dist[j]):
					max_cnt = voting[j]
					min_dist = sum_of_dist[j]
					max_idx = j
				# print "max_cnt,max_idx : ", max_cnt,max_idx

			assert(not max_idx == -1)

			confuseMat[i][max_idx] += 1
			TestY.append(i)
			PredY.append(max_idx)

	print(clfr(TestY, PredY))

	print "Finish Knn ..."
	print "K_value: ", K_value
	return confuseMat




if __name__ == '__main__':
	# print Knn_KL(12)
	print Kmeans_KL([4,1,1,2],inner = "Kmeans",n_init = 5)
