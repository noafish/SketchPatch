import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os
from . import util


class HTML:
	"""This HTML class allows us to save images and write texts into a single HTML file.

	 It consists of functions such as <add_header> (add a text header to the HTML file),
	 <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
	 It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
	"""

	def __init__(self, web_dir, title, is_test=False, refresh=0):
	    """Initialize the HTML classes

	    Parameters:
	        web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
	        title (str)   -- the webpage name
	        refresh (int) -- how often the website refresh itself; if 0; no refreshing
	    """
	    self.title = title
	    self.web_dir = web_dir
	    self.img_dir = os.path.join(self.web_dir, 'images')
	    self.is_test = is_test
	    if not os.path.exists(self.web_dir):
	        os.makedirs(self.web_dir)
	    if not os.path.exists(self.img_dir):
	        os.makedirs(self.img_dir)

	    self.doc = dominate.document(title=title)
	    if refresh > 0:
	        with self.doc.head:
	            meta(http_equiv="refresh", content=str(refresh))

	def get_image_dir(self):
	    """Return the directory that stores images"""
	    return self.img_dir

	def add_header_old(self, text):
	    """Insert a header to the HTML file

	    Parameters:
	        text (str) -- the header text
	    """
	    with self.doc:
	        h3(text)

	def add_header(self, str):
	    if self.is_test:
	        with self.doc:
	            h3(str)
	    else:
	        with self.edoc:
	            h3(str)

	def add_images_old(self, ims, txts, links, width=400):
	    """add images to the HTML file

	    Parameters:
	        ims (str list)   -- a list of image paths
	        txts (str list)  -- a list of image names shown on the website
	        links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
	    """
	    self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
	    self.doc.add(self.t)
	    with self.t:
	        with tr():
	            for im, txt, link in zip(ims, txts, links):
	                with td(style="word-wrap: break-word;", halign="center", valign="top"):
	                    with p():
	                        with a(href=os.path.join('images', link)):
	                            img(style="width:%dpx" % width, src=os.path.join('images', im))
	                        br()
	                        p(txt)



	def add_images(self, epoch, ims, txts, links, width=400):
	    if epoch != -1:
	        img_dir_epoch = os.path.join(self.img_dir, str(epoch))
	        util.mkdirs(img_dir_epoch)
	    else:
	        img_dir_epoch = self.img_dir
	    
	    path_parts = img_dir_epoch.split('/')
	    
	    if self.is_test:
	        rel_path = path_parts[-1:]
	    else:
	        rel_path = path_parts[-2:]
	    rel_path = '/'.join(rel_path)
	    
	    self.add_table()
	    with self.t:
	        with tr():
	            for im, txt, link in zip(ims, txts, links):
	                with td(style="word-wrap: break-word;", halign="center", valign="top"):
	                    with p():
	                        with a(href=os.path.join(rel_path, link)):
	                            img(style="width:%dpx" % width, src=os.path.join(rel_path, im))
	                        br()
	                        p(txt)


	def save(self):
	    """save the current content to the HMTL file"""
	    html_file = '%s/index.html' % self.web_dir
	    f = open(html_file, 'wt')
	    f.write(self.doc.render())
	    f.close()

	def add_table(self, border=1):
	    self.t = table(border=border, style="table-layout: fixed;")
	    #self.doc.add(self.t)
	    if self.is_test:
	        self.doc.add(self.t)
	    else:
	        self.edoc.add(self.t)
	    

	def add_epoch_doc(self, epoch):
	    ttl = 'epoch %d' % epoch
	    self.edoc = dominate.document(title=ttl)

	def add_epoch_link(self, str, epoch):
	    with self.doc:
	        #h3(str)
	        nm = 'epoch %d' % epoch
	        a(nm, href=str, style="font-size:50px")
	        br()

	def save_top(self, epoch):
	    epoch_file = 'epoch_%d.html' % epoch
	    self.add_epoch_link(epoch_file, epoch)
	    html_file = '%s/index.html' % self.web_dir
	    f = open(html_file, 'wt')
	    f.write(self.doc.render())
	    f.close()

	def save_epoch(self, epoch):
	    html_file = '%s/epoch_%d.html' % (self.web_dir, epoch)
	    f = open(html_file, 'wt')
	    f.write(self.edoc.render())
	    f.close()


if __name__ == '__main__':  # we show an example usage here.
	html = HTML('web/', 'test_html')
	html.add_header('hello world')

	ims, txts, links = [], [], []
	for n in range(4):
	    ims.append('image_%d.png' % n)
	    txts.append('text_%d' % n)
	    links.append('image_%d.png' % n)
	html.add_images(ims, txts, links)
	html.save()
