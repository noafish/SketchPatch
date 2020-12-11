import numpy as np
import os
import ntpath
import time
from . import util
from . import html
import cv2 as cv
import re


# save image to the disk
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
	image_dir = webpage.get_image_dir()
	short_path = ntpath.basename(image_path[0])
	name = os.path.splitext(short_path)[0]
	webpage.add_header(name)
	ims, txts, links = [], [], []

	for label, image in list(visuals.items()):
	    im_data = image[0]

	#for label, im_data in visuals.items():
	    im = util.tensor2im(im_data)
	    image_name = '%s_%s.png' % (name, label)
	    save_path = os.path.join(image_dir, image_name)

	    h, w, _ = im.shape
	    if aspect_ratio > 1.0:
	        im = cv.resize(im, (h, int(w * aspect_ratio)), fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC) 
	    if aspect_ratio < 1.0:
	        im = cv.resize(im, (int(h / aspect_ratio), w), interpolation=cv.INTER_CUBIC) 
	    util.save_image(im, save_path)

	    ims.append(image_name)
	    txts.append(label)
	    links.append(image_name)
	webpage.add_images(-1, ims, txts, links, width=width)


class Visualizer():
	def __init__(self, opt):
	    self.display_id = opt.display_id
	    self.use_html = opt.isTrain and not opt.no_html
	    self.win_size = opt.display_winsize
	    self.name = opt.name
	    self.opt = opt
	    self.saved = False
	    self.curr_epoch = -1
	    if self.display_id > 0:
	        import visdom
	        self.ncols = opt.display_ncols
	        self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, raise_exceptions=True)
	        
	    if self.use_html:
	        self.web_dir = os.path.join(opt.save_dir, 'web')
	        self.webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
	        self.img_dir = os.path.join(self.web_dir, 'images')
	        print('create web directory %s...' % self.web_dir)
	        util.mkdirs([self.web_dir, self.img_dir])
	    self.log_name = os.path.join(opt.save_dir, 'loss_log.txt')
	    with open(self.log_name, "a") as log_file:
	        now = time.strftime("%c")
	        log_file.write('================ Training Loss (%s) ================\n' % now)

	def reset(self):
	    self.saved = False

	def throw_visdom_connection_error(self): 
	    print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
	    exit(1)

	# |visuals|: dictionary of images to display or save
	def display_current_results(self, visuals, epoch, itnum, save_result):

	    if self.display_id > 0:  # show images in the browser using visdom
	        ncols = self.ncols
	        if ncols > 0:        # show all the images in one visdom panel
	            ncols = min(ncols, len(visuals))
	            h, w = next(iter(visuals.values())).shape[:2]
	            table_css = """<style>
	                    table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
	                    table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
	                    </style>""" % (w, h)  # create a table css
	            # create a table of images.
	            title = self.name
	            label_html = ''
	            label_html_row = ''
	            images = []
	            idx = 0
	            for label, image in visuals.items():
	                image_numpy = util.tensor2im(image)
	                label_html_row += '<td>%s</td>' % label
	                images.append(image_numpy.transpose([2, 0, 1]))
	                idx += 1
	                if idx % ncols == 0:
	                    label_html += '<tr>%s</tr>' % label_html_row
	                    label_html_row = ''
	            white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
	            while idx % ncols != 0:
	                images.append(white_image)
	                label_html_row += '<td></td>'
	                idx += 1
	            if label_html_row != '':
	                label_html += '<tr>%s</tr>' % label_html_row
	            try:
	                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
	                                padding=2, opts=dict(title=title + ' images'))
	                label_html = '<table>%s</table>' % label_html
	                self.vis.text(table_css + label_html, win=self.display_id + 2,
	                              opts=dict(title=title + ' labels'))
	            except VisdomExceptionBase:
	                self.create_visdom_connections()

	        else:     # show each image in a separate visdom panel;
	            idx = 1
	            try:
	                for label, image in visuals.items():
	                    image_numpy = util.tensor2im(image)
	                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
	                                   win=self.display_id + idx)
	                    idx += 1
	            except VisdomExceptionBase:
	                self.create_visdom_connections()


	    if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
	        self.saved = True
	        if self.curr_epoch < epoch:
	            self.webpage.add_epoch_doc(epoch)
	            self.webpage.save_top(epoch)
	            self.curr_epoch = epoch
	        self.img_dir_epoch = os.path.join(self.img_dir, str(epoch))
	        util.mkdirs(self.img_dir_epoch)
	        # save images to the disk
	        for label, image in visuals.items():
	            image_numpy = util.tensor2im(image)
	            img_path = os.path.join(self.img_dir_epoch, 'epoch%.3d_%.4d_%s.png' % (epoch, itnum, label))
	            util.save_image(image_numpy, img_path)

	        # update website
	        self.webpage.add_header('epoch [%d], iter [%d]' % (epoch, itnum))
	        ims, txts, links = [], [], []

	        for label, image_numpy in visuals.items():
	            image_numpy = util.tensor2im(image)
	            img_path = 'epoch%.3d_%.4d_%s.png' % (epoch, itnum, label)
	            ims.append(img_path)
	            txts.append(label)
	            links.append(img_path)
	        self.webpage.add_images(epoch, ims, txts, links, width=self.win_size)
	        self.webpage.save_epoch(epoch)


	# losses: dictionary of error labels and values
	def plot_current_losses(self, epoch, counter_ratio, losses):
	    if not hasattr(self, 'plot_data'):
	        self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
	    self.plot_data['X'].append(epoch + counter_ratio)
	    self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
	    try:
	        self.vis.line(
	            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
	            Y=np.array(self.plot_data['Y']),
	            opts={
	                'title': self.name + ' loss over time',
	                'legend': self.plot_data['legend'],
	                'xlabel': 'epoch',
	                'ylabel': 'loss'},
	            win=self.display_id)
	    except ConnectionError:
	        self.throw_visdom_connection_error()

	# losses: same format as |losses| of plot_current_losses
	def print_current_losses(self, epoch, i, losses, t, t_data):
	    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
	    for k, v in losses.items():
	        message += '%s: %.3f ' % (k, v)

	    print(message)
	    with open(self.log_name, "a") as log_file:
	        log_file.write('%s\n' % message)
