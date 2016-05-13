import numpy as np
import h5py
from operator import itemgetter


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


def sort_genre(Dict):
	"""
		sort dictionary by values

	"""
	return sorted(Dict.items(), key=itemgetter(1), reverse=True)


def load(filename):
	f = file(filename, 'r+')
	result = eval(f.read())
	f.close()
	return result


def save(item, filename):
	f = file(filename, 'w+')
	f.write(str(item))
	f.close()


def abstract_title(filename):
	artist = ''
	album  = ''
	title = ''
	with h5py.File(filename, 'r+') as f:
		meta_list = f.get('metadata').items()
		songs = np.array(meta_list[4][1])

		artist = songs[0][9]
		album = songs[0][14]
		title = songs[0][18]

	return (artist, album, title)


def speed_feature(filename):
	total_time_sec = 0.0
	section_number = 0
	average_bpm = 0.0
	with h5py.File(filename, 'r+') as f:
		analysis_list = f.get('analysis').items()
		# time intervals
		bars_start = np.array(analysis_list[1][1])
		beats_start = np.array(analysis_list[3][1])
		sections_start = np.array(analysis_list[5][1])
		segments_start = np.array(analysis_list[11][1])

		total_time_sec = segments_start[-1]
		section_number = len(sections_start)
		if len(beats_start) == 0:
			average_bpm = 0
		else:
			average_bpm = 60*(len(beats_start)-1)/beats_start[-1]

		# print 'total_time_sec: ', total_time_sec
		# print 'section_number: ', section_number
		# print 'average_bpm: ', average_bpm

	return (total_time_sec, section_number, average_bpm)
