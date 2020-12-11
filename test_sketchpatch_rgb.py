import os
import sys
import utils
import glob
import numpy as np
import cv2 as cv
import math
from PIL import Image
import copy
import random
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, breadth_first_order
from pathlib import Path
import torch
from options.test_options import TestOptions
from models import create_model
from data.base_dataset import BaseDataset, get_transform



SAVE_DIR = sys.argv[1]
SKETCH_DIR = sys.argv[2]
OUT_DIR = sys.argv[3]
PATCHSIZE = int(sys.argv[4])
ERD_DIL = int(sys.argv[5])

sys.argv = [sys.argv[0]]


svp = Path(SAVE_DIR)
CHECKPOINTS_DIR = svp.parent

if not os.path.exists(OUT_DIR):
	os.makedirs(OUT_DIR)




def  get_corner_close_to_neigh(cx, cy, dx, dy, offset):

	if cx < dx:
		corner = [dx-offset, cy]

	if cx > dx:
		corner = [dx+offset, cy]

	if cy < dy:
		corner = [cx, dy-offset]

	if cy > dy:
		corner = [cx, dy+offset]

	return corner



def translate_patch(hybrid, cmask, patch_id, loc, model):

	transform = get_transform(opt)
	hybrid_img = utils.ndarrayToPilImage(hybrid)
	hybrid_img = transform(hybrid_img)
	model_input = {'unstyled': hybrid_img, 'hybrid': hybrid_img, 'mask': cmask}
	model.set_input(model_input)
	model.test()
	result = model.fake
	im = utils.tensor2im(result)
	cres = np.asarray(im)

	return cres


def sigmoid(x):
	y = np.zeros(len(x))
	for i in range(len(x)):
		y[i] = 1 / (1 + math.exp(-x[i]))
	return y



def blend_overlap(a, b):

	c = 0 * copy.deepcopy(a)

	shp = a.shape
	if shp[0] < shp[1]:
		sz = shp[0]
		sigmoid_ = sigmoid(np.arange(-1, 1, 1/(sz/2)))
		sig = sigmoid_.reshape((len(sigmoid_),1))
		sig2 = sig - sig[0,0]
		sig3 = sig2 / sig2[-1,0]
		alpha = np.repeat(sig3, repeats=shp[1], axis=1)
		alpha = np.reshape(alpha, (alpha.shape[0],alpha.shape[1],1))
		alpha = np.repeat(alpha, repeats=3, axis=2)
		c = a * (1.0 - alpha) + b * alpha
	else:
		sz = shp[1]
		sigmoid_ = sigmoid(np.arange(-1, 1, 1/(sz/2)))
		sig = sigmoid_.reshape((1, len(sigmoid_)))
		sig2 = sig - sig[0,0]
		sig3 = sig2 / sig2[0,-1]
		alpha = np.repeat(sig3, repeats=shp[0], axis=0)
		alpha = np.reshape(alpha, (alpha.shape[0],alpha.shape[1],1))
		alpha = np.repeat(alpha, repeats=3, axis=2)
		c = a * (1.0 - alpha) + b * alpha

	return c




def run_one_order(padded_input, order, adj, loc, all_patches, patch_size, model, context, final_outdir, style_name, disp_name, binary_thresh):

	styledResults = {}
	full_destyled = (patch_size - context) * (patch_size - context)

	nump = len(all_patches)
	rf_score = np.zeros((nump,1))

	h,w,c = padded_input.shape
	res = copy.deepcopy(padded_input)
	mask = np.ones([h, w])
	transform = get_transform(opt) #, grayscale=True)
	all_can = []
	all_can.append(copy.deepcopy(res))

	adj += adj.transpose()
	visited = np.zeros((adj.shape[0]), dtype=int)

	margin = []

	for idx,o in enumerate(order):

		if idx == 0:
			first = o

		visited[o] = 1
		tmp = adj[o,:]
		nidx = [x for x in range(len(tmp)) if tmp[x] > 0]
		noverlap = tmp[nidx]
		vnidx = visited[nidx]
		vnidx2 = [ix for ix,x in enumerate(vnidx) if x == 1]
		o_dep = np.asarray([nidx[index] for index in vnidx2])
		o_dep_overlap = np.asarray([noverlap[index] for index in vnidx2])

		curPatch = all_patches[o]
		cloc = loc[o]
		c = cloc

		hybrid = res[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size, :]
		cmask = mask[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size]
		
		msk = 255 - np.sum(hybrid,axis=2)/3
		msk = msk.astype(int)
		unstyled = msk[cmask == 1]
		nz = np.count_nonzero(unstyled)

		if nz == 0:
			margin = margin + [o]
			mask[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size] = -1
			continue

			
		cres = translate_patch(hybrid, cmask, o, c, model)

		x1,x2,y1,y2 = 0,0,0,0
		buf = round(context/2)
		buf2 = round(buf/2)
		
		for dp in o_dep:

			if dp == o-1:
				y1 = buf
				olreg1 = res[c[0]:c[0]+patch_size, c[1]:c[1]+context, :]
				olreg2 = cres[:, 0:context, :]
				blended = blend_overlap(olreg1, olreg2)
				cres[:, 0:context, :] = blended
			if dp == o+1:
				y2 = -buf
				olreg2 = res[c[0]:c[0]+patch_size, c[1]+patch_size-context:c[1]+patch_size, :]
				olreg1 = cres[:, patch_size-context:patch_size, :]
				blended = blend_overlap(olreg1, olreg2)
				cres[:, patch_size-context:patch_size, :] = blended
			if dp < o-1:
				x1 = buf
				olreg1 = res[c[0]:c[0]+context, c[1]:c[1]+patch_size, :]
				olreg2 = cres[0:context, :, :]
				blended = blend_overlap(olreg1, olreg2)
				cres[0:context, :, :] = blended
			if dp > o+1:
				x2 = -buf
				olreg2 = res[c[0]+patch_size-context:c[0]+patch_size, c[1]:c[1]+patch_size, :]
				olreg1 = cres[patch_size-context:patch_size, :, :]
				blended = blend_overlap(olreg1, olreg2)
				cres[patch_size-context:patch_size, :, :] = blended



		res[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size] = cres
		mask[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size] = -1

		all_can.append(copy.deepcopy(res))



	name_str = '%s_%s_%d' % (style_name, disp_name, first)
	file_name = os.path.join(final_outdir, name_str+'.png')
	samename = glob.glob(file_name)
	currnum = str(len(samename)+1)
	sname = os.path.join(final_outdir, name_str+'_'+currnum+'.png')
	print(sname)

	utils.save_image(res, sname)

	#all_can.append(copy.deepcopy(res))
	#all_can.append(copy.deepcopy(res))
	#all_can.append(copy.deepcopy(res))
	#utils.gen_gif(all_can, sname.replace('.png', '.gif'), 3, 1)



def run_one_sketch(sketch, model, style_name, mode='', num_iter=3):
	
	im_name = os.path.join(SKETCH_DIR, sketch)
	p = sketch.split('.')
	disp_name = '.'.join(p[:-1])
	
	final_outdir = OUT_DIR
	final_outdir = os.path.join(final_outdir, style_name)
	if not os.path.exists(final_outdir):
		os.makedirs(final_outdir)

	patch_size = PATCHSIZE
	context = int(patch_size / 4)
	binary_thresh = 175

	im = utils.load_image_rgb(im_name)
	if mode == 'erode':
		im = utils.adjust_thickness(im, 1, num_iter)
		disp_name = disp_name + '_erd_%d' % num_iter
	if mode == 'dilate':
		im = utils.adjust_thickness(im, 0, num_iter)
		disp_name = disp_name + '_dil_%d' % num_iter
	all_patches, patch_nz, loc, num_patch_rows, num_patch_cols, padded_input = utils.image_to_patches(im, context, patch_size, binary_thresh)
	adj, non_empty = utils.calc_adj(all_patches, context, num_patch_rows, num_patch_cols)
	adjt = copy.deepcopy(adj)
	full_overlap = context * patch_size

	adjt = np.asarray(adjt)
	adjt[:,2] = full_overlap - adjt[:,2] 
	numpa = len(all_patches)
	adj2 = coo_matrix((adjt[:,2], (adjt[:,0], adjt[:,1])), shape=(numpa, numpa))
	mst = minimum_spanning_tree(adj2)
	ncomp, lbls = connected_components(mst)
	adj = np.asarray(adj)
	adj = coo_matrix((adj[:,2], (adj[:,0], adj[:,1])), shape=(numpa, numpa))
	adj_mat = adj.toarray()
	lbls2 = lbls[non_empty]
	comps, counts = np.unique(lbls2, return_counts=True)

	fullpatch = patch_size ** 2

	num_to_gen = 1

	for i in range(num_to_gen):
		
		order = []
		dep = {}

		for idx,c in enumerate(comps):
			
			comp = c
			members = [x for x,y in enumerate(lbls) if y == comp]
			valid_members = [m for m in members if patch_nz[m] < 0.67 * fullpatch and patch_nz[m] > 0.1 * fullpatch]
			if len(valid_members) < 1:
				continue
			start = random.randrange(0,len(valid_members))
			start = valid_members[start]
			
			corder, pred = breadth_first_order(mst, start, directed=False)
			order.extend(corder)
		
		run_one_order(padded_input, order, adj_mat, loc, all_patches, patch_size, model, context, final_outdir, style_name, disp_name, binary_thresh)
		

	
def run_one_model(topt, save_dir, erd_dil):
	
	style_name = os.path.basename(save_dir)
	topt.opt.save_dir = style_name
	opt = topt.post_parse()
	
	opt.nThreads = 1
	opt.batchSize = 1
	opt.no_flip = True
	opt.display_id = -1
	
	opt.model = 'sketchpatch'
	opt.no_dropout = True
	opt.load_size = PATCHSIZE
	opt.crop_size = PATCHSIZE
	opt.input_nc = 3
	opt.output_nc = 3
	opt.isTrain = False

	torch.manual_seed(0)

	model = create_model(opt)
	if opt.eval:
		model.eval()
	model.setup(opt)
	

	sketches = [f for f in os.listdir(SKETCH_DIR) if f.lower().endswith(('.png','.jpg','jpeg'))]
	
	for sk in sketches:
		
		if erd_dil == 0:
			run_one_sketch(sk, model, style_name, '')
		

		if erd_dil < 0:
			try:
				run_one_sketch(sk, model, style_name, mode='erode', num_iter=-1*erd_dil)
			except:
				print("failed on erosion with num iter %d" % (-1*erd_dil))

		if erd_dil > 0:
			try:
				run_one_sketch(sk, model, style_name, mode='dilate', num_iter=erd_dil)
			except:
				print("failed on dilation with num iter %d" % erd_dil)
		




if __name__ == '__main__':

	topt = TestOptions()
	opt = topt.parse()
	topt.opt.checkpoints_dir = CHECKPOINTS_DIR
	run_one_model(topt, SAVE_DIR, ERD_DIL)
	
