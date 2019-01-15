import numpy as np
import random as rng
import theano as th

def patchExtractor(kSize,imgs,pStride):
	print imgs.shape
	xsize = imgs.shape[0]
	ysize = imgs.shape[1]
	NChannelsPerImage = imgs.shape[2]
	
	N = kSize**2*NChannelsPerImage
    
	nPatches = int(np.floor((xsize-kSize)/pStride +1))*int(np.floor((ysize-kSize)/pStride +1)) 
    
	imagePatches = []

	for i in xrange(0,xsize,pStride):
		for j in xrange(0,ysize,pStride):
			if (kSize/2 < i <= xsize - (kSize/2+1)) & (kSize/2 < j <= ysize - (kSize/2+1)):
				imagePatches.append((imgs[i - kSize/2 : i + (kSize/2 + 1), j - kSize/2 : j + (kSize/2 + 1),:]).reshape(1,N))

	random.seed(1)
	random.shuffle(imagePatches) 
    
	return  imagePatches 
