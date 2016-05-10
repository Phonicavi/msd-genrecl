import os
# import keras
import numpy as np
import h5py

rootDir = '/Users/Phonic/Desktop/MillionSongSubset/data'


def traverse():
	count = 0
	for root, dirs, files in os.walk(rootDir):
		for filespath in files:
			count += 1
			filename = os.path.join(root, filespath)
			print(filename)
			structure(filename)

	return count


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
	with h5py.File(filename, 'r') as f:

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
					# print(data.shape)
					# print(data.size)
					# print(data.dtype)
					# print
					print('Contents: \n')
					print(npdata)
					print
				pass

		if flag:
			print('-------- End --------')
			print
		pass



def main():
	total = traverse()
	print(total)

	structure('./test_data/TRAAADZ128F9348C2E.h5', True)
	print



if __name__ == '__main__':
	main()


