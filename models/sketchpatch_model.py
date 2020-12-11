import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import os
from util import util


class SketchPatchModel(BaseModel):

	@staticmethod
	def modify_commandline_options(parser, is_train=True):

		parser.set_defaults(no_dropout=True)
		return parser

	def __init__(self, opt):
		
		BaseModel.__init__(self, opt)
		self.loss_names = ['G_R', 'G_B', 'G_D', 'D']
		self.visual_names = ['styled', 'unstyled', 'hybrid', 'fake', 'mask', 'smooth_fake', 'smooth_styled']

		if self.isTrain:
		    self.model_names = ['G_A', 'D_A']
		else:
		    self.model_names = ['G_A']

		self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
										not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


		if self.isTrain:
			self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
			
			self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
			self.criterionCycle = torch.nn.L1Loss()
			self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
			
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)

			self.smoothing = networks.GaussianSmoothing(opt.input_nc, 10, 10).to(self.device)
			
			self.w_recon = opt.w_recon
			self.w_blur = opt.w_blur
			self.w_disc = opt.w_disc
			
			self.loss_D = 0
			self.do_blur = True
			self.input_nc = opt.input_nc
			


	def set_input(self, input):

		if self.isTrain:
			self.styled = input['styled'].to(self.device)
			self.unstyled = input['unstyled'].to(self.device)
			self.hybrid = input['hybrid'].to(self.device)
			self.mask = input['mask'].unsqueeze(1).type('torch.FloatTensor').to(self.device)
			self.real = input['real'].to(self.device)
		else:
			self.unstyled = input['unstyled'].to(self.device)
			self.hybrid = input['hybrid'].unsqueeze(0).to(self.device)
			self.mask = torch.from_numpy(input['mask']).float().unsqueeze(0).unsqueeze(1).to(self.device)

		self.input = self.hybrid
		


	def forward(self):
		self.fake = self.netG_A(self.input)

	def forward_D(self):
		criterionGAN = networks.GANLoss('lsgan').to(self.device)
		pred = self.netD_A(self.fake)
		return criterionGAN(pred, True)

	def backward_D(self):
		pred_real = self.netD_A(self.real)
		loss_D_real = self.criterionGAN(pred_real, True)
		pred_fake = self.netD_A(self.fake.detach())
		loss_D_fake = self.criterionGAN(pred_fake, False)
		self.loss_D = (loss_D_real + loss_D_fake) * 0.5
		self.loss_D.backward()

	# def getInverseContextMask(self, contextMask):
	# 	inverseContextMask = contextMask.clone()
	# 	inverseContextMask[contextMask == 0] = 1
	# 	inverseContextMask[contextMask == 1] = 0
	# 	return inverseContextMask

	def set_do_blur(self, dblr):
		self.do_blur = dblr

	def backward_G(self):
		
		if self.w_blur > 0 and self.do_blur:
			sfake = self.smoothing(self.fake)
			sstyled = self.smoothing(self.styled)
			self.loss_G_B = self.w_blur * self.criterionCycle(sfake, sstyled)
			self.smooth_fake = sfake
			self.smooth_styled = sstyled
		else:
			self.loss_G_B = 0

		if self.w_disc > 0:
			self.loss_G_D = self.w_disc * self.criterionGAN(self.netD_A(self.fake), True)
		else:
			self.loss_G_D = 0

		
		if self.w_recon > 0:
			self.loss_G_R = self.w_recon * self.criterionCycle(self.fake, self.styled)
		else:
			self.loss_G_R = 0
		
		self.loss_G = self.loss_G_R + self.loss_G_D + self.loss_G_B
		self.loss_G.backward()

	def optimize_parameters(self):
		
		self.forward()
		self.set_requires_grad(self.netD_A, False)
		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()
		
		if self.w_disc > 0:
			self.set_requires_grad(self.netD_A, True)
			self.optimizer_D.zero_grad()
			self.backward_D()
			self.optimizer_D.step()

