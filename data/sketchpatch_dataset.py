import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from random import randrange
import numpy as np
from scipy import ndimage
import copy
import torch


class SketchPatchDataset(BaseDataset):
	"""
	This dataset class can load unaligned/unpaired datasets.

	It requires two directories to host training images from domain A '/path/to/data/trainA'
	and from domain B '/path/to/data/trainB' respectively.
	You can train the model with the dataset flag '--dataroot /path/to/data'.
	Similarly, you need to prepare two directories:
	'/path/to/data/testA' and '/path/to/data/testB' during test time.
	"""

	def __init__(self, opt):
		"""Initialize this dataset class.

		Parameters:
			opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		BaseDataset.__init__(self, opt)
		self.dir_styled = os.path.join(opt.dataroot)  # create a path '/path/to/data/trainA'
		self.styled_paths = sorted(make_dataset(self.dir_styled, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
		self.styled_size = len(self.styled_paths)  # get the size of dataset A

		self.input_nc = self.opt.input_nc       # get the number of channels of input image
		self.output_nc = self.opt.output_nc      # get the number of channels of output image
		if self.input_nc == 1:
			self.transform = get_transform(self.opt, grayscale=True)
		else:
			self.transform = get_transform(self.opt)

		self.imsize = opt.crop_size



	def getInverseContextMask(self, contextMask):
		inverseContextMask = contextMask.copy()
		inverseContextMask[contextMask == 0] = 1
		inverseContextMask[contextMask == 1] = 0
		return inverseContextMask


	def createHybrid(self, unstyled, styled, contextMask):
		imsize = unstyled.shape[0]
		contextMask[contextMask == -1] = 0
		inverseContextMask = self.getInverseContextMask(contextMask)
		if self.input_nc > 1:
			tmp1 = np.zeros((imsize,imsize,self.input_nc))
			tmp2 = copy.deepcopy(tmp1)
			for ic in range(self.input_nc):
				tmp1[:,:,ic] = contextMask
				tmp2[:,:,ic] = inverseContextMask
			contextMask = tmp1
			inverseContextMask = tmp2

		hybrid = styled * inverseContextMask + unstyled * contextMask
		return hybrid

	def pilImageToNdarray(self, pilIm):
		return np.asarray(pilIm)

	def ndarrayToPilImage(self, arr):
		return Image.fromarray(np.uint8(arr))

	def rreplace(self, s, old, new, count):
		return (s[::-1].replace(old[::-1], new[::-1], count))[::-1]

	def __getitem__(self, index):
		"""Return a data point and its metadata information.

		Parameters:
			index (int)      -- a random integer for data indexing

		Returns a dictionary that contains A, B, A_paths and B_paths
			A (tensor)       -- an image in the input domain
			B (tensor)       -- its corresponding image in the target domain
			A_paths (str)    -- image paths
			B_paths (str)    -- image paths
		"""

		styled_path = self.styled_paths[index % self.styled_size] 
		unstyled_path = self.rreplace(styled_path, "styled", "plain", 1)
		unstyled_path = self.rreplace(unstyled_path, "real_A", "fake_B", 1)
		if self.input_nc == 1:
			styled_img = Image.open(styled_path).convert('L')
			unstyled_img = Image.open(unstyled_path).convert('L')
		else:
			styled_img = Image.open(styled_path)
			unstyled_img = Image.open(unstyled_path)
		
		maskDirections = ["Top", "Bot", "Left", "Right"]
		dirs = np.round(np.random.rand(4))
		dirs = np.nonzero(dirs)
		dirs = dirs[0].tolist()
		cdirs = [maskDirections[i] for i in dirs]

		all_plain_prob = 0.5  #0.1

		all_plain = random.random()
		if all_plain < all_plain_prob:
			cdirs = []

		random.shuffle(cdirs)

		styled = self.pilImageToNdarray(styled_img)
		unstyled = self.pilImageToNdarray(unstyled_img)
		
		vert0 = styled.shape[0]
		hori0 = styled.shape[1]
		vert = vert0
		hori = hori0

		min_ol = int(vert / 16)
		max_ol = int(vert / 2)

		mask = np.ones([vert0, hori0])
		mask_loss = np.zeros([vert0, hori0])
		buff = int(vert / 8)

		for d in cdirs:
			if d == "Top":
				top = random.randint(min_ol, max_ol)
				vert = vert - top
				mask[0:top, :] = -1
				top2 = top + buff
				top2 = min(top2, vert0)
				mask_loss[0:top2, :] = 1

			if d == "Bot":
				bot = random.randint(min_ol, max_ol)
				vert = vert - bot
				mask[vert0-bot:vert0, :] = -1
				bot2 = bot + buff
				bot2 = min(bot2, vert0)
				mask_loss[vert0-bot2:vert0, :] = 1

			if d == "Left":
				left = random.randint(min_ol, max_ol)
				hori = hori - left
				mask[:, 0:left] = -1
				left2 = left + buff
				left2 = min(left2, hori0)
				mask_loss[:, 0:left2] = 1

			if d == "Right":
				right = random.randint(min_ol, max_ol)
				hori = hori - right
				mask[:, hori0-right:hori0] = -1
				right2 = right + buff
				right2 = min(right2, hori0)
				mask_loss[:, hori0-right2:hori0] = 1


		flip = random.random()
		if flip < 1/3:
			styled = np.flipud(styled)
			unstyled = np.flipud(unstyled)
		if flip >= 1/3 and flip < 2/3:
			styled = np.fliplr(styled)
			unstyled = np.fliplr(unstyled)

		styled_img = self.ndarrayToPilImage(styled)
		unstyled_img = self.ndarrayToPilImage(unstyled)

		hybrid_img = self.createHybrid(unstyled, styled, mask)
		hybrid_img = self.ndarrayToPilImage(hybrid_img)

		styled = self.transform(styled_img)
		unstyled = self.transform(unstyled_img)
		hybrid = self.transform(hybrid_img)

		rreal = random.randrange(0, self.styled_size-1)
		real_path = self.styled_paths[rreal]
		if self.input_nc == 1:
			real_img = Image.open(real_path).convert('L')
		else:
			real_img = Image.open(real_path)

		if flip < 1/3:
			real_img = real_img.transpose(Image.FLIP_TOP_BOTTOM)
		if flip >= 1/3 and flip < 2/3:
			real_img = real_img.transpose(Image.FLIP_LEFT_RIGHT)

		real_img = np.asarray(real_img)
		real_img = self.ndarrayToPilImage(real_img)
		real = self.transform(real_img)
		
		context_mask = mask

		return {'styled': styled, 'unstyled': unstyled, 'hybrid': hybrid, 'mask': context_mask, 'styled_path': styled_path, 'unstyled_path': unstyled_path, 'real': real}


	def __len__(self):
		"""Return the total number of images in the dataset.

		As we have two datasets with potentially different number of images,
		we take a maximum of
		"""
		return self.styled_size
