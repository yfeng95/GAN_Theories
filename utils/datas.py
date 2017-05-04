import os,sys
from PIL import Image
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

prefix = './Datas/'

def get_img(img_path, is_crop=True, crop_h=256, resize_h=64):
	img=scipy.misc.imread(img_path).astype(np.float)
	resize_w = resize_h
	if is_crop:
		crop_w = crop_h
		h, w = img.shape[:2]
		j = int(round((h - crop_h)/2.))
		i = int(round((w - crop_w)/2.))
		cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])
	else:
		cropped_image = scipy.misc.imresize(img,[resize_h, resize_w])
	return np.array(cropped_image)/255.0


class celebA():
	def __init__(self):
		datapath = prefix + 'celebA'
		self.z_dim = 100
		self.size = 64
		self.channel = 3
		self.data = glob(os.path.join(datapath, '*.jpg'))

		self.batch_count = 0

	def __call__(self,batch_size):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-2:
			self.batch_count += 1
		else:
			self.batch_count = 0

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

		batch = [get_img(img_path, True, 128, self.size) for img_path in path_list]
		batch_imgs = np.array(batch).astype(np.float32)
		
		return batch_imgs

	def data2fig(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig

class cifar():
	def __init__(self):
		datapath = prefix + 'cifar10'
		self.z_dim = 100
		self.size = 64
		self.channel = 3
		self.data = glob(os.path.join(datapath, '*'))

		self.batch_count = 0

	def __call__(self,batch_size):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-2:
			self.batch_count += 1
		else:
			self.batch_count = 0

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

		batch = [get_img(img_path, False, 128, self.size) for img_path in path_list]
		batch_imgs = np.array(batch).astype(np.float32)
	
		return batch_imgs

	def data2fig(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig


class mnist():
	def __init__(self):
		datapath = prefix + 'mnist'
		self.z_dim = 100
		self.size = 64
		self.channel = 1
		self.data = input_data.read_data_sets(datapath, one_hot=True)

	def __call__(self,batch_size):
		batch_imgs = np.zeros([batch_size, self.size, self.size, self.channel])

		batch_x,y = self.data.train.next_batch(batch_size)
		batch_x = np.reshape(batch_x, (batch_size, 28, 28, self.channel))
		for i in range(batch_size):
			img = batch_x[i,:,:,0]
			batch_imgs[i,:,:,0] = scipy.misc.imresize(img, [self.size, self.size])
		batch_imgs /= 255.
		return batch_imgs, y

	def data2fig(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig	


if __name__ == '__main__':
	data = mnist()
	imgs,_ = data(20)

	fig = mnist.data2fig(imgs[:16,:,:])
	plt.savefig('Samples/test.png', bbox_inches='tight')
	plt.close(fig)
