# -*- coding:utf-8 -*-  
import numpy as np
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFECV, f_classif,chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from minepy import MINE
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold

def featureSelection (TrainSet,TestSet,method = 'mean',testmode = False,n_features_to_select = None):
	assert testmode in [False,True],'TestMode must be Boolean!'
	assert method in ['RFECV','f_class','MIC','RFC','Stab','mean','Ridge'],'Not this method!'
	print 'Using Feature Selection Method: ',method
	X = []
	Y = []

	for gen in range (len(TrainSet)):
		for songs in TrainSet[gen]:
			X.append(songs)
			Y.append(gen)
	X = np.array(X)
	Y = np.array(Y)

	select_idx = []

	names = [i for i in range(0,len(TrainSet[0][0]))]

	ranks = {}
	def rank_to_dict(ranks, names, order=1):
	    minmax = MinMaxScaler()
	    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
	    ranks = map(lambda x: round(x, 3), ranks)
	    return dict(zip(names, ranks ))

	if not method == 'RFECV' and n_features_to_select == None:
		##  select half of the original feature
		n_features_to_select = len(TrainSet[0][0])/2

	print 'Start Selection ... '


	# if method == 'RFECV' or method == 'mean':
	# 	# svc = SVC(kernel="linear")
	# 	ri =  Ridge(alpha=7)

	# 	rfecv = RFECV(estimator=ri, step=1, verbose = 0,
	# 		cv=3,
 #            # scoring='accuracy',
 #              )
	# 	rfecv.fit(X,Y)
	# 	ranks["RFECV"] = rank_to_dict(map(float, rfecv.ranking_), names, order=-1)
	# 	print("<RFECV> Optimal number of features  : %d" % rfecv.n_features_)
	# 	if method == 'RFECV':
	# 		for (idx,b) in enumerate(rfecv.support_):
	# 			if (b == True):
	# 				select_idx.append(idx)
			# print select_idx
		# print [i for (i,b) in rfecv.support_ == True) 

	# if method == 'Ridge' or method == 'mean':
	# 	ridge = Ridge(alpha=7)
	# 	ridge.fit(X, Y)
	# 	ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)


	
	# if method == 'MIC' or method == 'mean':
	# 	mine = MINE()
	# 	mic_scores = []
	# 	for i in range(X.shape[1]):
	# 	    mine.compute_score(X[:,i], Y)
	# 	    m = mine.mic()
	# 	    mic_scores.append(m)
	# 	ranks["MIC"] = rank_to_dict(mic_scores, names)

	if method == 'f_class' or method == 'mean':
		f, pval  = f_classif(X, Y)
		ranks["f_class"] = rank_to_dict(f, names)
	if method == 'f_class' or method == 'mean':
		f, pval  = f_classif(X, Y)
		ranks["f_class1"] = rank_to_dict(f, names)

	# if method == 'chi2' or method == 'mean':
	# 	f, pval  = chi2(X, Y)
	# 	ranks["chi2"] = rank_to_dict(f, names)

	if method == 'RFC' or method == 'mean':
		rf = RandomForestClassifier(criterion = 'entropy', n_estimators=200)
		rf.fit(X,Y)
		ranks["RFC"] = rank_to_dict(rf.feature_importances_, names)

	r = {}
	for name in names:
	    r[name] = round(np.mean([ranks[me][name] 
	                             for me in ranks.keys()]), 3)

	methods = sorted(ranks.keys())
	ranks["mean"] = r
	methods.append("mean")

	# print ranks['mean']

	if testmode:
		print "\t%s" % "\t\t".join(methods)
		for name in names:
		    print "%s\t%s" % (name, "\t\t".join(map(str, 
		                         [ranks[method][name] for method in methods])))



	if not method == 'RFECV':
		dic = ranks[method]
		# print dic
		print '===='
		select_idx = [tp[0] for tp in sorted(dic.iteritems(), key=lambda d:d[1], reverse = True)[:n_features_to_select]]

	# print 'select_idx: ',select_idx

	for gen in range(len(TrainSet)):
		for (i,songs) in enumerate(TrainSet[gen]):
			newsongs = []
			for idx in range(len(songs)):
				if idx in select_idx:
					newsongs.append(songs[idx])
			TrainSet[gen][i] = newsongs
		for (i,songs) in enumerate(TestSet[gen]):
			newsongs = []
			for idx in range(len(songs)):
				if idx in select_idx:
					newsongs.append(songs[idx])
			TestSet[gen][i] = newsongs

	print 'len of Left Feature: ',len(TrainSet[0][0])
	print 'Finish Selection ... '
	return TrainSet,TestSet










