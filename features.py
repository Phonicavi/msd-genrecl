import os
import numpy as np
from matplotlib import pyplot
import h5py
import helper as hp


rootDir = '/Users/Phonic/Desktop/MillionSongSubset/data'

class Features(object):
	"""docstring for Features"""
	NAME = "features"
	def __init__(self, arg):
		super(Features, self).__init__()
		self.arg = arg


def traverse(I2S):
	count = 0
	for root, dirs, files in os.walk(rootDir):
		for filespath in files:
			filename = os.path.join(root, filespath)
			div = filename.split('/')
			ID = div[len(div)-1][:-3]
			if I2S.has_key(ID): # consider
				count += 1
				(artist, album, title) = hp.abstract_title(filename)
				del I2S[ID]
				I2S[filename] = [artist, album, title]
			pass
	print 'new added: ', count

	return I2S


def segments(filename):
	# index 6 - 12
	with h5py.File(filename, 'r+') as f:
		analysis_list = f.get('analysis').items()
		# draw terms as nparray
		'''
			for i in xrange(6,13):
				terms = np.array(analysis_list[i][1])
				print terms.shape
				print type(terms)

		'''
		pitches = np.array(analysis_list[10][1])
		timbre = np.array(analysis_list[12][1])
		print pitches[1:10][:]


def speed_allocate(inventory):
	tag = hp.load('my_tag.dic')
	Time_Sec = {}
	Section_Number = {}
	Average_BPM = {}
	for filename in inventory.keys():
		div = filename.split('/')
		ID = div[len(div)-1][:-3]
		genr = tag[ID]
		# print genr
		(total_time_sec, section_number, average_bpm) = hp.speed_feature(filename)
		if not Time_Sec.has_key(genr):
			Time_Sec[genr] = []
			Section_Number[genr] = []
			Average_BPM[genr] = []
		Time_Sec[genr].append(total_time_sec)
		Section_Number[genr].append(section_number)
		Average_BPM[genr].append(average_bpm)

	return (Time_Sec, Section_Number, Average_BPM)


def plot_distribution(splist):
	pyplot.figure()
	row = len(splist)
	base = row*100 + 10
	color = { 'Rock':'red', 'Rap':'green', 'Latin':'yellow', 'Country':'blue', 'Electronic':'purple' }

	for genr in splist.keys():
		base += 1
		ax = pyplot.subplot(base)
		ax.set_title(genr)
		pyplot.hist(splist[genr], 100, normed=1, facecolor=color[genr], alpha=0.5)

	pyplot.show()



def main():
	# new_i2s = hp.load('new_i2s.dic')
	# filename_i2s = traverse(new_i2s)
	filename_i2s = hp.load('filename_i2s.dic')

	# hp.save(filename_i2s, 'filename_i2s.dic')
	# print filename_i2s
	# print len(filename_i2s)

	# segments('TRAAAAW128F429D538.h5')

	print '--- speed collection ---'

	'''
	(Time_Sec, Section_Number, Average_BPM) = speed_allocate(filename_i2s)

	hp.save(Time_Sec, 'Time_Sec.dic')
	hp.save(Section_Number, 'Section_Number.dic')
	hp.save(Average_BPM, 'Average_BPM.dic')

	'''

	TS = hp.load('Time_Sec.dic')
	SN = hp.load('Section_Number.dic')
	ABPM = hp.load('Average_BPM.dic')

	plot_distribution(TS)
	plot_distribution(SN)
	plot_distribution(ABPM)



if __name__ == '__main__':
	main()

