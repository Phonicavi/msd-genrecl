import os
# import keras
import numpy as np
# import pandas as pd
import h5py
import helper as hp


rootDir = '/Users/Phonic/Desktop/MillionSongSubset/data'


def traverse():
	count = 0
	General = {'country':0, 'hip hop':0, 'metal':0, 'jazz':0, 'pop':0}
	Inventory = []
	for root, dirs, files in os.walk(rootDir):
		for filespath in files:
			filename = os.path.join(root, filespath)
			# print(filename)
			# structure(filename)
			(flag, genre) = extract(filename)
			if flag:
				count += 1
				General[genre] += 1
				tupple = (filename, genre)
				Inventory.append(tupple)
				# 
				with h5py.File(filename, 'r+') as f:
					info = str(np.array(f.get('metadata').items()[4][1])[0])
					if info.find("Dive") >= 0:
						print(filename)
						print(info)
						print
						print(str(np.array(f.get('metadata').items()[0][1])))
						print
				# 
				# structure(filename)
			pass
			# print


	return (count, General, Inventory)


def structure(filename, flag = False):
	"""
		Try to understand the inner structure in these 3 tables

		analysis: 16

			0:  <HDF5 dataset "bars_confidence": shape (291,), type "<f8">
			1:  <HDF5 dataset "bars_start": shape (291,), type "<f8">
			2:  <HDF5 dataset "beats_confidence": shape (291,), type "<f8">
			3:  <HDF5 dataset "beats_start": shape (291,), type "<f8">
			4:  <HDF5 dataset "sections_confidence": shape (8,), type "<f8">
			5:  <HDF5 dataset "sections_start": shape (8,), type "<f8">
			6:  <HDF5 dataset "segments_confidence": shape (562,), type "<f8">
			7:  <HDF5 dataset "segments_loudness_max": shape (562,), type "<f8">
			8:  <HDF5 dataset "segments_loudness_max_time": shape (562,), type "<f8">
			9:  <HDF5 dataset "segments_loudness_start": shape (562,), type "<f8">
			10: <HDF5 dataset "segments_pitches": shape (562, 12), type "<f8">
			11: <HDF5 dataset "segments_start": shape (562,), type "<f8">
			12: <HDF5 dataset "segments_timbre": shape (562, 12), type "<f8">
			13: <HDF5 dataset "songs": shape (1,), type "|V220">
			14: <HDF5 dataset "tatums_confidence": shape (582,), type "<f8">
			15: <HDF5 dataset "tatums_start": shape (582,), type "<f8">


		metadata: 5

			0:	<HDF5 dataset "artist_terms": shape (10,), type "|S256">
			1:	<HDF5 dataset "artist_terms_freq": shape (10,), type "<f8">
			2:	<HDF5 dataset "artist_terms_weight": shape (10,), type "<f8">
			3:	<HDF5 dataset "similar_artists": shape (100,), type "|S20">
			5:	<HDF5 dataset "songs": shape (1,), type "|V5320">

		
		musicbrainz: 3

			0:	<HDF5 dataset "artist_mbtags": shape (0,), type "|S256">
			1:	<HDF5 dataset "artist_mbtags_count": shape (0,), type "<i4">
			2:	<HDF5 dataset "songs": shape (1,), type "|V8">

	"""
	with h5py.File(filename, 'r+') as f:

		if flag:
			print('---'+filename+'---')
		pass

		Table = []
		# 3 tables in Table
		# analysis, metadata, musicbrainz
		for tables in f.keys():
			tab = f.get(tables)
			Table.append(tab)

			if flag:
				print('Loading... ')
				print(tab) # class 'h5py._hl.group.Group'
				print(' ...\t')
			pass

			for entry in tab.items():
				# print(entry) # type 'tuple', length = 2
				data = entry[1]
				npdata = np.array(data)
				if flag:
					print(data) # class 'h5py._hl.dataset.Dataset'
					'''
						print(data.shape)
						print(data.size)
						print(data.dtype)
						print

					'''
					print('Contents: \n')
					print(npdata)
					print
				pass

		if flag:
			print('-------- End --------')
			print
		pass


def extract(filename):
	"""
		Extract necessary titles

	"""
	with h5py.File(filename, 'r+') as f:

		Keys = ['country', 'hip hop', 'metal', 'jazz', 'pop']
		# print('---'+filename+'---')
		
		meta_list = f.get('metadata').items()
		# draw terms as nparray
		terms = np.array(meta_list[0][1])
		terms_freq = np.array(meta_list[1][1])
		terms_weight = np.array(meta_list[2][1])
		if terms.size == 0:
			return (False, '')
		else:
			'''
				print('genre = '+str(terms[0]))
				print('freq = '+str(terms_freq[0]))
				print('weight = '+str(terms_weight[0]))

			'''
			genre = str(terms[0:3])
			return_label = ''
			# collect genre
			for G_label in Keys:
				if genre.find(G_label) >= 0:
					return_label = G_label
					break

			if return_label == '':
				return (False, '')
			else:
				return (True, return_label)
		pass
	pass



def main():
	
	(total, General, Inventory) = traverse()
	print(total)

	
	# General = hp.load('genre.txt')
	hp.save(General, 'genre.txt')
	hp.save(Inventory, 'inventory.txt')
	raw_genre_list = hp.sort_genre(General)

	print(raw_genre_list)
	

	# test-part
	# traverse()
	# structure('TRAAADZ128F9348C2E.h5', True)
	# print(extract('TRAAAAW128F429D538.h5'))
	print



if __name__ == '__main__':
	main()


