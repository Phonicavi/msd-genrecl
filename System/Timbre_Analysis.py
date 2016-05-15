# -*- coding:utf-8 -*-  

from BasicClass import *
import sys
import random
from sklearn.cluster import KMeans
from sklearn import mixture

NUM_NEED_PER_GENRE = [200,200,200,200]
GENRES = ['Jazz','Rap','Rock','Country']



def Inner_Kmeans(SingleGenreSongs,numOfClusters):
	means = [item.mean_timbreVec for item in SingleGenreSongs]
	means_array = np.array(means)
	model = KMeans(n_clusters = numOfClusters,n_init = 8)
	model.fit(means)

	lbls = model.labels_

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
	model = mixture.GMM(n_components = numOfClusters,covariance_type = 'full',n_init = 8)
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
	iter_num = 15
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



	## Using inner_kmeaans select cluster centroid  
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

		# for idx in range(len(allGenreSongs[i])):
		# 	cluster_centroids[i].addMean_timbre(allGenreSongs[i][idx].mean_timbreVec)   
		# 	cluster_centroids[i].addCov_timbre(allGenreSongs[i][idx].cov_timbreVec)
		# cluster_centroids[i].averMean_timbre(len(allGenreSongs[i]))
		# cluster_centroids[i].averCov_timbre(len(allGenreSongs[i]))
	print "len of cluster_centroids", len(cluster_centroids)

	for it in range(iter_num):
		print "#iter",it,"..."
		num_in_centroid = [0 for i in range(len(cluster_centroids))]
		new_cluster_centroids = []
		for i in range (len(cluster_centroids)):
			new_cluster_centroids.append(Centroid_Kmeans(str(i)))
		for song in allSongs:
			KL_dist = []
			for centroid in cluster_centroids:
				KL_dist.append(getKLdiv(song.mean_timbreVec,centroid.mean_timbreVec,song.cov_timbreVec,centroid.cov_timbreVec))	# pay attention to something need to be discarded
			val,idx = min((val,idx) for (idx,val) in enumerate(KL_dist))
			num_in_centroid[idx] += 1
			new_cluster_centroids[idx].addMean_timbre(song.mean_timbreVec)
			new_cluster_centroids[idx].addCov_timbre(song.cov_timbreVec)

		for i in range(len(new_cluster_centroids)):
			if (num_in_centroid[i] > 0):
				print "# in cluster ",i,": ",num_in_centroid[i]
				new_cluster_centroids[i].averMean_timbre(num_in_centroid[i])
				new_cluster_centroids[i].averCov_timbre(num_in_centroid[i])
		cluster_centroids = deepcopy(new_cluster_centroids)

	return cluster_centroids


def Kmeans_KL(cluster_num_of_each_genre,inner = "Kmeans"):
	## FetchData
	NeedReFetch = False
	allGenreSongsTrain,allGenreSongsTest = fetchData_TA(NUM_NEED_PER_GENRE,GENRES,NeedReFetch)

	print "Start Kmeans training ..."

	assign_cluster = [[] for i in range(len(GENRES))]

	idx = 0;
	for gen in range(len(GENRES)):
		for n in range(cluster_num_of_each_genre[gen]):
			assign_cluster[gen].append(idx)
			idx += 1


	print assign_cluster
	# assign_cluster = [[0,1,2],[3],[4],[5]]

	model = Kmeans_KL_train(allGenreSongsTrain,len(GENRES),assign_cluster,inner)

	print "Finish Kmeans training ..."

	confuseMat = [[0 for i in range(len(GENRES))] for j in range(len(GENRES))];

	print "Start Kmeans predicting ..."
	cnt = 0;
	
	for i in range(len(GENRES)):
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
	print "Start Kmeans predicting ..."
	return confuseMat

def Knn_KL(K_value):
	## FetchData
	NeedReFetch = False
	allGenreSongsTrain,allGenreSongsTest = fetchData_TA(NUM_NEED_PER_GENRE,GENRES,NeedReFetch)

	confuseMat = [[0 for i in range(len(GENRES))] for j in range(len(GENRES))];

	print "Start Knn ..."

	cnt = 0;

	for i in range(len(GENRES)):
		for testSong in allGenreSongsTest[i]:
			cnt += 1
			print "Dealing with testSongs ", cnt
			k_nearest = [[sys.maxint,-1] for j in range(K_value)]		## (KL_dist,cluster_centroid)

			for j in range(len(GENRES)):
				for trainSong in allGenreSongsTrain[j]:
					tp,idx = max((tp,idx) for (idx,tp) in enumerate(k_nearest))
					cen_idx = k_nearest[idx][1]
					tmp_dis = getKLdiv(testSong.mean_timbreVec,trainSong.mean_timbreVec,testSong.cov_timbreVec,trainSong.cov_timbreVec)

					if  tmp_dis < k_nearest[idx][0]:
						k_nearest[idx][0] = tmp_dis
						k_nearest[idx][1] = j
			voting = [0 for j in range(len(GENRES))]
			sum_of_dist = [0.0 for j in range(len(GENRES))]

			for j in range(K_value):
				voting[k_nearest[j][1]]+=1
				sum_of_dist[k_nearest[j][1]] += k_nearest[j][0]

			max_cnt = -1;
			min_dist = sys.maxint
			max_idx = -1;

			for j in range(len(GENRES)):
				if (voting[j] > max_cnt) or (voting[j] == max_cnt and min_dist > sum_of_dist[j]):
					max_cnt = voting[j]
					min_dist = sum_of_dist[j]
					max_idx = j
				# print "max_cnt,max_idx : ", max_cnt,max_idx

			assert(not max_idx == -1)

			confuseMat[i][max_idx] += 1
	print "Finish Knn ..."
	print "K_value: ", K_value
	return confuseMat




if __name__ == '__main__':
	# print Knn_KL(12)
	print Kmeans_KL([3,1,1,1],inner = "Kmeans")
