import os
import sys
import utils
import glob
import numpy as np
import cv2 as cv
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
ERD_DIL = int(sys.argv[4])

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

	transform = get_transform(opt, grayscale=True)

	hybrid_img = utils.ndarrayToPilImage(hybrid)
	hybrid_img = transform(hybrid_img)
	model_input = {'unstyled': hybrid_img, 'hybrid': hybrid_img, 'mask': cmask}
	model.set_input(model_input)
	model.test()

	result = model.fake
	im = utils.tensor2im(result)
	cres = np.asarray(im)

	return cres



def run_one_order(padded_input, order, adj, loc, all_patches, patch_size, model, context, final_outdir, style_name, disp_name, binary_thresh):

	styledResults = {}

	full_destyled = (patch_size - context) * (patch_size - context)

	nump = len(all_patches)
	rf_score = np.zeros((nump,1))

	h,w = padded_input.shape
	res = copy.deepcopy(padded_input)
	mask = np.ones([h, w])
	transform = get_transform(opt, grayscale=True)
	all_can = []
	all_can.append(copy.deepcopy(255-res))

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

		hybrid = res[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size]
		cmask = mask[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size]
		
		unstyled = hybrid[cmask == 1]
		nz = np.count_nonzero(unstyled)

		if nz == 0:
			margin = margin + [o]
			mask[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size] = -1
			continue

		if nz / full_destyled > 0.95:

			if len(o_dep) > 0:

				od = o_dep_overlap.argmax()
				od = o_dep[od]
				hybrid2 = copy.deepcopy(hybrid)
				c2 = copy.deepcopy(c)

				if od == o-1:
					hybrid2[:,patch_size-context:patch_size] = 0
					c2[1] = c[1] + context
				if od == o+1:
					hybrid2[:,0:context] = 0
					c2[1] = c[1] - context
				if od < o-1:
					hybrid2[patch_size-context:patch_size,:] = 0
					c2[0] = c[0] + context
				if od > o+1:
					hybrid2[0:context,:] = 0
					c2[0] = c[0] - context

				cres = translate_patch(hybrid2, cmask, o, c, model)
				
				x1,x2,y1,y2 = 0,0,0,0
				buf = round(context/2)
				buf2 = 6

				for dp in o_dep:
					if dp == o-1:
						y1 = buf
					if dp == o+1:
						y2 = -buf
					if dp < o-1:
						x1 = buf
					if dp > o+1:
						x2 = -buf

				if od == o-1:
					y2 = min(y2, -(context+buf2))
				if od == o+1:
					y1 = max(y1, context+buf2)
				if od < o-1:
					x2 = min(x2, -(context+buf2))
				if od > o+1:
					x1 = max(x1, context+buf2)

				res[c[0]+x1:c[0]+patch_size+x2, c[1]+y1:c[1]+patch_size+y2] = cres[x1:patch_size+x2, y1:patch_size+y2]

				htemp = res[c2[0]:c2[0]+patch_size, c2[1]:c2[1]+patch_size]
				hybrid3 = copy.deepcopy(htemp)

				if od == o-1:
					hybrid3[:,patch_size-context+buf2:patch_size] = 0
				if od == o+1:
					hybrid3[:,0:context-buf2] = 0
				if od < o-1:
					hybrid3[patch_size-context+buf2:patch_size,:] = 0
				if od > o+1:
					hybrid3[0:context-buf2,:] = 0

				cres = translate_patch(hybrid3, cmask, o, c, model)


				x1,x2,y1,y2 = 0,0,0,0

				for dp in o_dep:
					if dp == o-1:
						y1 = buf
					if dp == o+1:
						y2 = -buf
					if dp < o-1:
						x1 = buf
					if dp > o+1:
						x2 = -buf

				if od == o-1:
					y2 = min(y2, -(context+buf2))
				if od == o+1:
					y1 = max(y1, context+buf2)
				if od < o-1:
					x2 = min(x2, -(context+buf2))
				if od > o+1:
					x1 = max(x1, context+buf2)

				res[c2[0]+x1:c2[0]+patch_size+x2, c2[1]+y1:c2[1]+patch_size+y2] = cres[x1:patch_size+x2, y1:patch_size+y2]

				all_can.append(copy.deepcopy(255-res))


		cres = translate_patch(hybrid, cmask, o, c, model)

		x1,x2,y1,y2 = 0,0,0,0
		buf = round(context/2)

		for dp in o_dep:
			if dp == o-1:
				y1 = buf
			if dp == o+1:
				y2 = -buf
			if dp < o-1:
				x1 = buf
			if dp > o+1:
				x2 = -buf


		res[c[0]+x1:c[0]+patch_size+x2, c[1]+y1:c[1]+patch_size+y2] = cres[x1:patch_size+x2, y1:patch_size+y2]
		mask[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size] = -1

		all_can.append(copy.deepcopy(255-res))



	for m in margin:

		tmp = adj[m,:]

		nidx = [x for x in range(len(tmp)) if tmp[x] > 0]
		noverlap = tmp[nidx]
		vnidx = visited[nidx]
		vnidx2 = [ix for ix,x in enumerate(vnidx) if x == 1]
		m_dep = np.asarray([nidx[index] for index in vnidx2])
		m_dep_overlap = np.asarray([noverlap[index] for index in vnidx2])

		for dix,dp in enumerate(m_dep):

			olap = m_dep_overlap[dix]
			if olap < 100: 
				continue
			cloc = loc[m]
			dloc = loc[dp]
			cx = cloc[0]
			cy = cloc[1]
			dx = dloc[0]
			dy = dloc[1]

			corner = get_corner_close_to_neigh(cx, cy, dx, dy, context)
			c = corner
			hybrid = res[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size]
			cmask = mask[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size]
			unstyled = hybrid[cmask == 1]

			cres = translate_patch(hybrid, cmask, m, c, model)

			x1,x2,y1,y2 = 0,0,0,0
			buf = round(context/2)
			buf2 = round((patch_size - context) / 2)
			
			for dp2 in m_dep:
				cbuf = buf
				if dp2 == dp:
					cbuf = buf2

				if dp2 == o-1:
					y1 = cbuf
				if dp2 == o+1:
					y2 = -cbuf
				if dp2 < o-1:
					x1 = cbuf
				if dp2 > o+1:
					x2 = -cbuf

			res[c[0]+x1:c[0]+patch_size+x2, c[1]+y1:c[1]+patch_size+y2] = cres[x1:patch_size+x2, y1:patch_size+y2]
			mask[c[0]:c[0]+patch_size, c[1]:c[1]+patch_size] = -1

			all_can.append(copy.deepcopy(255-res))


	name_str = '%s_%s_%d' % (style_name, disp_name, first)
	file_name = os.path.join(final_outdir, name_str+'.png')
	samename = glob.glob(file_name)
	currnum = str(len(samename)+1)
	sname = os.path.join(final_outdir, name_str+'_'+currnum+'.png')
	print(sname)

	h,w = res.shape
	top, bot, left, right = utils.crop_image_by_bb(res, binary_thresh)
	pad = 4
	top = top - pad
	bot = bot + pad
	left = left - pad
	right = right + pad
	top = max(0, top)
	bot = min(h, bot)
	left = max(0, left)
	right = min(w, right)
	res = res[top:bot, left:right]

	utils.save_image(255-res, sname)
	utils.gen_gif(all_can, sname.replace('.png', '.gif'), 20, 0)



def run_one_sketch(sketch, model, style_name, mode='', num_iter=3):
	
	im_name = os.path.join(SKETCH_DIR, sketch)
	p = sketch.split('.')
	disp_name = '.'.join(p[:-1])
	
	final_outdir = OUT_DIR
	final_outdir = os.path.join(final_outdir, style_name)
	if not os.path.exists(final_outdir):
		os.makedirs(final_outdir)

	patch_size = 64
	context = int(patch_size/4)
	binary_thresh = 175

	im = utils.load_image(im_name)
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

			corder, pred = breadth_first_order(mst, start, directed=False)#62)
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
	opt.model =  'sketchpatch'
	opt.no_dropout = True
	opt.load_size = 64
	opt.crop_size = 64
	opt.input_nc = 1
	opt.output_nc = 1

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
	
