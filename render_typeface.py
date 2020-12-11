
import sys
import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2 as cv
from scipy import ndimage


#### Instructions: ####
# This script receives three parameters:
# font_dir = a folder containing a set of .ttf files
# out_dir = the output folder, which will contain a subfolder per .ttf file
# mode = which characters to render:
#    0 = uppercase letters + digits
#    1 = uppercase + lowercase letters
#    2 = uppercase letters


letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
letters2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

RES = 400

font_dir = sys.argv[1]
out_dir = sys.argv[2]
mode = int(sys.argv[3])


REF = 'reference_roboto_reg_1000'


def draw_character(fnt, c, font_sz):
	im = Image.new('RGB', (RES,RES), (0,0,0))
	draw = ImageDraw.Draw(im)
	font = ImageFont.truetype(fnt, font_sz)
	draw.text((0, 0), c, font=font)
	return im

def bbox(img):
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]

	return rmin, rmax, cmin, cmax


def crop_image_by_bb(im):
	binaryThresh = 50
	thresh1, binaryImage = cv.threshold(im, binaryThresh, 255, cv.THRESH_BINARY)
	top, bot, left, right = bbox(binaryImage)

	return im[top:bot, left:right]


def compare_size(fnt, c, fsz, shp):
	im = draw_character(fnt, c, fsz)
	im = np.asarray(im)
	im = crop_image_by_bb(im)
	shp1 = im.shape
	diff = abs(shp[0] - shp1[0])

	return im, diff




if __name__ == '__main__':

	ref_o = os.path.join(REF, 'O.png')
	ref_s = os.path.join(REF, 'S.png')
	ref_g = os.path.join(REF, 'G.png')
	ref_i = os.path.join(REF, 'I.png')
	ref_4 = os.path.join(REF, '4.png')

	ref_o = crop_image_by_bb(np.asarray(Image.open(ref_o)))
	ref_s = crop_image_by_bb(np.asarray(Image.open(ref_s)))
	ref_g = crop_image_by_bb(np.asarray(Image.open(ref_g)))
	ref_i = crop_image_by_bb(np.asarray(Image.open(ref_i)))
	ref_4 = crop_image_by_bb(np.asarray(Image.open(ref_4)))

	shp_o = ref_o.shape
	shp_s = ref_s.shape
	shp_g = ref_g.shape
	shp_i = ref_i.shape
	shp_4 = ref_4.shape

	fonts = os.listdir(font_dir)
	
	if mode == 0:
		all_chars = letters + digits

	if mode == 1:
		all_chars = letters + letters2

	if mode == 2:
		all_chars = letters

	
	for f in fonts:
		fnt_nm = f.split('.')[0]
		odir = os.path.join(out_dir, fnt_nm)
		fnt = os.path.join(font_dir, f)
		
		if not os.path.exists(odir):
			os.makedirs(odir)
		else:
			continue
			
		
		print(f)

		szdict = {}

		for fsz in range(100,401):

			if fsz%10 != 0:
				continue


			diff_o, diff_s, diff_g, diff_i = 0,0,0,0

			im, diff_o = compare_size(fnt, 'O', fsz, shp_o)
			im, diff_s = compare_size(fnt, 'S', fsz, shp_s)
			im, diff_g = compare_size(fnt, 'G', fsz, shp_g)
			im, diff_i = compare_size(fnt, 'I', fsz, shp_i)

			diff = (diff_o + diff_s + diff_g + diff_i) / 4

			szdict[fsz] = diff


		
		best_fsz = min(szdict, key=szdict.get)
		#print("best font size: ", best_fsz)
		#print(szdict[best_fsz])

		

		for c in all_chars:

			im = draw_character(fnt, c, best_fsz)
			im = np.asarray(im)
			im = im[:,:,0]
			im = Image.fromarray(im)
			fname = os.path.join(odir, c+'.png')
			im.save(fname)

