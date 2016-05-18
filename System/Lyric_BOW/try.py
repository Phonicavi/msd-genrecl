import numpy as np 
from sklearn.decomposition import PCA,SparsePCA
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter 
from sklearn import preprocessing
from sklearn.externals import joblib

from sklearn.externals  import joblib
LyricDic = (joblib.load('ori_BOW_data_for_each_song.pkl'))
with open('my_tag_for_lyric.dic','r') as f:
	tagDic = eval(f.read())

C = []
B = []
# CNT = []

for gen in ['Rock','Rap','Country','Electronic','Latin']:
	all_feas  =[]
	cnt = 0
	for item in LyricDic.keys():
		if (tagDic[item] == gen):
			all_feas.append(LyricDic[item])
			cnt += 1



	# all_feas = [dic[item] for ]

	print len(all_feas)
	b = np.array(all_feas)
	c = b.astype(bool).astype(int)
	C.append(sum(c)/float(cnt))
	B.append(sum(b)/float(cnt))

	# print sum(b)
	# print sum(c)

# min_max_scaler = preprocessing.StandardScaler()
# B = min_max_scaler.fit_transform(B)
B = np.array(B).T 
C = np.array(C).T 

# np.var(B,1).shape


head = 0
tail = 5000
x = [i for i in range(head,tail)]

# print np.var(B)

varB = [(idx,item) for (idx,item) in enumerate(list(np.var(B,1)))]
sort_varB = sorted(varB,key=itemgetter(1),reverse = True)[:500]

varC = [(idx,item) for (idx,item) in enumerate(list(np.var(C,1)))]
sort_varC = sorted(varC,key=itemgetter(1),reverse = True)[:500]

# plotnum = 2000
# splot = plt.subplot(2,1,1)
# plt.plot(x[:plotnum],[item[1] for item in sort_varB[:plotnum]])
# splot = plt.subplot(2,1,2)
# plt.plot(x[:plotnum],[item[1] for item in sort_varC[:plotnum]])


sort_varB = [idx for (idx,item) in sort_varB]
sort_varC = [idx for (idx,item) in sort_varC]

joblib.dump(sort_varC,'select_word_idx_list.pkl',compress = 3)

print sort_varB
print sort_varC


# print varB

B = [0 for i in range(5000)]
C = [0 for i in range(5000)]



for i in range(5000):
	if i not in sort_varB:
		B[i] = 0
	else:
		B[i] = varB[i][1]
	if i not in sort_varC:
		C[i] = 0
	else:
		C[i] = varC[i][1]
# print B

# print len(sum(b))
# plotnum = 5000
# splot = plt.subplot(2,1,1)
# plt.bar(x[:plotnum],B[:plotnum])
# splot = plt.subplot(2,1,2)
# plt.bar(x[:plotnum],C[:plotnum])

plt.show()

# print all_feas
# pca = PCA(n_components = 100,n_jobs = 4)
# new_feas = pca.fit_transform(np.array(all_feas))
# with open('try.dat','w+') as f:
# 	f.write(str(new_feas))
