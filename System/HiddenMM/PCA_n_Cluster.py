# -*- coding:utf-8 -*-  

import numpy as np 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GMM,DPGMM
import random
import os
import h5py
from sklearn.externals  import joblib
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt 

NUM_NEED_PER_GENRE = [200,200,200,200,200,200,200]
GENRES = ['Jazz','Rap','Rock','Country','Blues','Latin','Electronic']
USED_GENRES = [1,0,1,0,1,0,0]

# FETCH_WHAT = 'segments_pitches'
FETCH_WHAT = 'segments_timbre'

def get_all_feas():
	ALL_FEA = []

	if False:

		rootDir = "../../fliter_h5_data/"
		for (idx,gen) in enumerate(GENRES):
			if (USED_GENRES[idx] == 0):
				continue
			print 'fetching ', gen, '...'
			for filename in os.listdir(os.path.join(rootDir,gen)):
				# print filename
				f = h5py.File(os.path.join(rootDir,gen,filename))
				ALL_FEA += list(f['analysis'][FETCH_WHAT])

		print len(ALL_FEA)

		joblib.dump(ALL_FEA,'ALL_FEA.dat',compress = 3)

		# with open('ALL_FEA.dat',"w+") as f1:
		# 	f1.write(str(ALL_FEA))

	else:
		print 'Loading feas ...'
		ALL_FEA = joblib.load('ALL_FEA.dat')
		# with open('ALL_FEA.dat','r') as f1:
			# ALL_FEA = eval(f1.read())


	return random.sample(ALL_FEA,int(len(ALL_FEA)*0.1))



def PCA_n_plot():
	all_feas = get_all_feas();


	print 'Start PCA ...'
	pca = PCA(n_components = 3,whiten = False)
	new_feas = pca.fit_transform(np.array(all_feas))
	Fig = plt.figure()
	ax = Axes3D(Fig)
	print 'Start Ploting'
	ax.scatter(new_feas[:,0],new_feas[:,1],new_feas[:,2])
	plt.show()

	
def cluster_n_plot():
	all_feas = get_all_feas();
	print 'Kmeans... '
	x = []
	y = []
	
	for num_cluster in range(6,20):
		est = KMeans(n_clusters = num_cluster)
		est.fit(all_feas)
		print num_cluster,est.inertia_/len(all_feas)
		x.append(num_cluster)
		y.append(est.inertia_/len(all_feas))
	plt.plot(x[:],y[:])
	plt.show()










if __name__ == '__main__':
	cluster_n_plot()
	




