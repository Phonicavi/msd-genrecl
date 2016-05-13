import os
import numpy as np
import h5py
import helper as hp


rootDir = '/Users/Phonic/Desktop/MillionSongSubset/data'

def traverse(I2S, Tag):
	count = 0
	for root, dirs, files in os.walk(rootDir):
		for filespath in files:
			filename = os.path.join(root, filespath)
			div = filename.split('/')
			ID = div[len(div)-1][:-3]
			if Tag.has_key(ID): # consider
				if I2S.has_key(ID): # already done
					pass
				else: # do something
					count += 1
					(artist, album, title) = hp.abstract_title(filename)
					I2S[ID] = [artist, title]
			pass
	print 'new added: ', count

	return I2S


i2s = file('filter_ID_SONG.dic', 'r')
id_song = eval(i2s.read())
i2s.close()
print 'already has: ', len(id_song)

my_tag = file('my_tag.dic', 'r')
tag = eval(my_tag.read())
my_tag.close()
print 'total has: ', len(tag)


new_i2s = traverse(id_song, tag)

print 'new total has: ', len(new_i2s)
hp.save(new_i2s, 'new_i2s.dic')


