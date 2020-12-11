from .base_options import BaseOptions
import torch


class TestOptions(BaseOptions):
	"""This class includes test options.

	It also includes shared options defined in BaseOptions.
	"""

	def initialize(self, parser):
	    parser = BaseOptions.initialize(self, parser)  # define shared options
	    parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
	    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
	    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
	    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
	    # Dropout and Batchnorm has different behavioir during training and test.
	    parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
	    parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
	    # rewrite devalue values
	    parser.set_defaults(model='test')
	    # To avoid cropping, the load_size should be the same as crop_size
	    parser.set_defaults(load_size=parser.get_default('crop_size'))
	    self.isTrain = False
	    
	    return parser


	def parse(self):

	    opt = self.gather_options()
	    opt.isTrain = False

	    # process opt.suffix
	    if opt.suffix:
	        suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
	        opt.name = opt.name + suffix
	        
	    self.opt = opt
	    
	    # set gpu ids
	    str_ids = self.opt.gpu_ids.split(',')
	    self.opt.gpu_ids = []
	    for str_id in str_ids:
	        id = int(str_id)
	        if id >= 0:
	            self.opt.gpu_ids.append(id)
	    if len(self.opt.gpu_ids) > 0:
	        torch.cuda.set_device(self.opt.gpu_ids[0])
	    
	    return self.opt


	def post_parse(self):
		
	    self.print_options(self.opt)
	    return self.opt
