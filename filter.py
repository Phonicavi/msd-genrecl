import os
import numpy as np
import h5py
import helper as hp


rootDir = '/Users/Phonic/Desktop/MillionSongSubset/data'

def traverse(Inventory, Tag):
	count = 0
	Genre = {}
	Store = []
	for root, dirs, files in os.walk(rootDir):
		for filespath in files:
			filename = os.path.join(root, filespath)
			div = filename.split('/')
			ID = div[len(div)-1][:-3]
			if Inventory.has_key(ID):
				genr = Inventory[ID]
				if Genre.has_key(genr):
					Genre[genr] += 1
				else:
					Genre[genr] = 1
				tupple = (filename, genr)
				Store.append(tupple)
			count += 1
			pass

	return (count, Genre, Store)


def joint(inv1, inv2):
	storage = {}
	count = 0
	for (item, genr) in inv1:
		if storage.has_key(item):
			pass
		else:
			storage[item] = 1
			if genr == 'Latin':
				count += 1

	for (item, genr) in inv2:
		if storage.has_key(item):
			pass
		else:
			storage[item] = 1
			if genr == 'Latin':
				count += 1

	print 'Latin:', count

	return storage


def tag(target):
	INVENTORY = {}
	TAG = {}
	filename = '/Users/Phonic/Desktop/MillionSongSubset/GenreTagsOfMSD/'+target
	f = file(filename, 'r+')
	while True:
		line = f.readline().split()
		if not line:
			break;
		if len(line) >= 2:
			name = line[0]
			genr = line[1]
			INVENTORY[name] = genr
			if TAG.has_key(genr):
				TAG[genr] += 1
			else:
				TAG[genr] = 1
	f.close()

	return (INVENTORY, TAG)



def main():
	# print '-----'
	# (Inventory, Tag) = tag('tag.txt')
	# print len(Inventory)
	# print len(Tag)
	# # print Inventory
	# print hp.sort_genre(Tag)
	# print
	# (count, Genre, Store) = traverse(Inventory, Tag)
	# print count
	# print len(Genre)
	# print len(Store)
	# print hp.sort_genre(Genre)
	# print
	# print


	print '-----'
	(Inventory, Tag) = tag('tag1.txt')
	# print len(Inventory)
	# print len(Tag)
	# print Inventory
	# print hp.sort_genre(Tag)
	print
	(count, Genre, Store) = traverse(Inventory, Tag)
	print count
	print len(Genre)
	print len(Store)
	print hp.sort_genre(Genre)
	print
	print

	inv1 = Store

	print '-----'
	(Inventory, Tag) = tag('tag2.txt')
	# print len(Inventory)
	# print len(Tag)
	# print Inventory
	# print hp.sort_genre(Tag)
	print
	(count, Genre, Store) = traverse(Inventory, Tag)
	print count
	print len(Genre)
	print len(Store)
	print hp.sort_genre(Genre)
	print
	print

	inv2 = Store


	print len(joint(inv1, inv2))


if __name__ == '__main__':
	main()