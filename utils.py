import torch
import os
import cv2 as cv
import numpy as np
from PIL import Image
import copy
from scipy import ndimage
import imageio



def bbox(img):
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]

	return rmin, rmax, cmin, cmax


def crop_image_by_bb(im, binary_thresh):
	thresh1, binaryImage = cv.threshold(im, binary_thresh, 255, cv.THRESH_BINARY)
	top, bot, left, right = bbox(binaryImage)

	return top, bot, left, right


def load_image(im_name):
	im = np.asarray(Image.open(im_name))
	if len(im.shape) == 3:
		im = np.asarray(Image.open(im_name).convert('L'))
	im = 255 - im
	top, bot, left, right = crop_image_by_bb(im, 50)
	croppedIm = im[top:bot, left:right]
	
	padsz = 40
	padded = np.pad(croppedIm, ((padsz, padsz),(padsz, padsz)), 'constant')
	
	return padded

def load_image_rgb(im_name):
	im0 = Image.open(im_name)
	im = np.asarray(im0)
	
	if len(im.shape) == 2:
		im0 = im0.convert('RGB')
		im = np.asarray(im0)

	msk = np.sum(im,axis=2) / 3
	msk = 255-msk
	top, bot, left, right = crop_image_by_bb(msk, 50)
	croppedIm = im[top:bot, left:right, :]
	
	padsz = 40
	padded = np.pad(croppedIm, ((padsz, padsz),(padsz, padsz),(0,0)), 'constant', constant_values=255)
	
	return padded
	

def image_to_patches(im, context, patch_size, binary_thresh):#, imDebug, binary_thresh, isSketch):
	
	nonOverlapRegionSize = patch_size - context
	

	if len(im.shape) > 2:
		msk = np.sum(im,axis=2) / 3
		msk = 255-msk
		top, bot, left, right = crop_image_by_bb(msk, binary_thresh)
		croppedIm = im[top:bot, left:right, :]
		padsz = 40
		croppedIm = np.pad(croppedIm, ((padsz, padsz),(padsz, padsz), (0,0)), 'constant', constant_values=255)
		h, w, c = croppedIm.shape
		padSizeRight = patch_size - (w - int(w / nonOverlapRegionSize)*nonOverlapRegionSize)
		padSizeBot = patch_size - (h - int(h / nonOverlapRegionSize)*nonOverlapRegionSize)
		padded = np.pad(croppedIm, ((0, padSizeBot),(0, padSizeRight), (0,0)), 'constant', constant_values=255)
		ph, pw, pc = padded.shape
		is_rgb = 1
	else:
		top, bot, left, right = crop_image_by_bb(im, binary_thresh)
		croppedIm = im[top:bot, left:right]
		padsz = 40
		croppedIm = np.pad(croppedIm, ((padsz, padsz),(padsz, padsz)), 'constant')
		h, w = croppedIm.shape
		padSizeRight = patch_size - (w - int(w / nonOverlapRegionSize)*nonOverlapRegionSize)
		padSizeBot = patch_size - (h - int(h / nonOverlapRegionSize)*nonOverlapRegionSize)
		padded = np.pad(croppedIm, ((0, padSizeBot),(0, padSizeRight)), 'constant')
		ph, pw = padded.shape
		is_rgb = 0


	num_patch_cols = int((pw-patch_size) / nonOverlapRegionSize + 1)
	num_patch_rows = int((ph-patch_size) / nonOverlapRegionSize + 1)

	line_color = (255, 255, 255)
	all_patches = []
	patch_nz = []
	curW = 0
	curH = 0
	
	loc = []
	
	for i in range(0, num_patch_rows):
		for j in range(0, num_patch_cols):

			patchTop = curH
			patchBottom = curH + patch_size - 1
			patchLeft = curW
			patchRight = curW + patch_size - 1

			p1 = (patchLeft, patchTop)
			p2 = (patchLeft, patchBottom)
			p3 = (patchRight, patchBottom)
			p4 = (patchRight, patchTop)

			if is_rgb:
				patch = padded[patchTop:(patchBottom+1), patchLeft:(patchRight+1), :]
				msk = patch[:,:,0]
				msk = 255 - msk
				nz = np.count_nonzero(msk)
			else:
				patch = padded[patchTop:(patchBottom+1), patchLeft:(patchRight+1)]
				nz = np.count_nonzero(patch)
			
			patch_nz.append(nz)
			loc.append([patchTop,patchLeft])
			
			all_patches.append(patch)
			curW = curW + nonOverlapRegionSize
			
		curW = 0
		curH = curH + nonOverlapRegionSize

	return all_patches, patch_nz, loc, num_patch_rows, num_patch_cols, padded



def are_neigh(p1, context, direction):
	if len(p1.shape) > 2:
		hIm, wIm, cIm = p1.shape
		is_rgb = 1
	else:
		hIm, wIm = p1.shape
		is_rgb = 0
	if (direction == 'Top'):
		if is_rgb:
			p12 = p1[:,:,0]
			p12 = 255 - p12
			nz = np.nonzero(p12[hIm-context:hIm,:])
		else:
			nz = np.nonzero(p1[hIm-context:hIm,:])
	elif (direction == 'Bot'):
		if is_rgb:
			p12 = p1[:,:,0]
			p12 = 255 - p12
			nz = np.nonzero(p12[0:context,:])
		else:
			nz = np.nonzero(p1[0:context,:])
	elif (direction == 'Left'):
		if is_rgb:
			p12 = p1[:,:,0]
			p12 = 255 - p12
			nz = np.nonzero(p12[:,wIm-context:wIm])
		else:
			nz = np.nonzero(p1[:,wIm-context:wIm])
	else: #direction == Right
		if is_rgb:
			p12 = p1[:,:,0]
			p12 = 255 - p12
			nz = np.nonzero(p12[:,0:context])
		else:
			nz = np.nonzero(p1[:,0:context])

	return nz[0].size


def calc_adj(all_patches, context, rows, cols):
	adj = []
	patch_size = all_patches[0].shape[0]
	full_overlap = context * patch_size
	non_empty = []
	for i in range(0, rows):
		for j in range(0, cols):
			patchIdx = cols*i + j
			curPatch = all_patches[patchIdx]
			if len(curPatch.shape) > 2:
				h, w, c = curPatch.shape
				tmp = curPatch[:,:,0]
				tmp = 255 - tmp
				nz = np.nonzero(tmp)
			else:
				h, w = curPatch.shape
				nz = np.nonzero(curPatch)
			if nz[0].size > 0:
				non_empty.append(patchIdx)

			if (j < cols-1):
				adj_score = are_neigh(curPatch, context, 'Left')
				if adj_score > 0:
					adj.append([patchIdx, patchIdx+1, adj_score])
			if (i < rows-1):
				adj_score = are_neigh(curPatch, context, 'Top')
				if adj_score > 0:
					adj.append([patchIdx, patchIdx+cols, adj_score])
			
	
	return adj, non_empty



def adjust_thickness(im, erode, num_iter=3):

	im1 = copy.deepcopy(im)

	if len(im.shape) > 2:
		im1 = 255-im1[:,:,0]

	if erode:
		im2 = ndimage.binary_erosion(im1/255, iterations=num_iter).astype(im.dtype)
	else:
		im2 = ndimage.binary_dilation(im1/255, iterations=num_iter).astype(im.dtype)

	if len(im.shape) > 2:
		im2 = 1 - im2
		im[:,:,0] = im2 
		im[:,:,1] = im2
		im[:,:,2] = im2
		im2 = im

	return im2*255


def save_image(image_numpy, image_path):
	image_pil = Image.fromarray(image_numpy)
	if image_pil.mode != 'RGB':
		image_pil = image_pil.convert('RGB')
	image_pil.save(image_path)


def tensor2im(input_image, imtype=np.uint8):

	if not isinstance(input_image, np.ndarray):
		if isinstance(input_image, torch.Tensor):
			image_tensor = input_image.data
		else:
			return input_image
		sz = image_tensor.size()
		if len(sz) == 4:
			if sz[1] == 3:
				image_tensor = image_tensor[0]
			elif sz[1]== 1:
				image_tensor = image_tensor[0,0]
		image_numpy = image_tensor.cpu().float().numpy()
		if sz[1] == 3:
			image_numpy = np.transpose(image_numpy, (1, 2, 0))
		image_numpy = (image_numpy + 1.0) / 2.0 * 255.0
	else:
		image_numpy = input_image

	return image_numpy.astype(imtype)


def pilImageToNdarray(pilIm):
	return np.asarray(pilIm)


def ndarrayToPilImage(arr):
	return Image.fromarray(np.uint8(arr))


def gen_gif(images, outname, speed, loops=0):

	imageio.mimsave(outname, images, loop=loops, fps=speed)

