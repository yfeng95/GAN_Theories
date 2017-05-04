import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os,sys

sys.path.append('utils')
from nets import *
from datas import *

def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])

class EBGAN():
	def __init__(self, generator, discriminator, data):
		self.generator = generator
		self.discriminator = discriminator
		self.data = data

		# data
		self.z_dim = self.data.z_dim
		self.size = self.data.size
		self.channel = self.data.channel

		self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

		# ebgan parameters
		margin = 50. #

		# nets
		self.G_sample = self.generator(self.z)

		self.D_real = self.discriminator(self.X)
		self.D_fake = self.discriminator(self.G_sample, reuse = True)
		
		# loss
		#L_real = tf.reduce_mean((self.X - self.D_real)**2, [1,2,3])
		#L_fake = tf.reduce_mean((self.G_sample - self.D_fake)**2, [1,2,3])
		L_real = tf.nn.l2_loss(self.X - self.D_real)
		L_fake = tf.nn.l2_loss(self.G_sample - self.D_fake)

		self.D_loss = L_real + tf.maximum(0., margin - L_fake)
		self.G_loss = L_fake
		
		# solver
		self.learning_rate = tf.placeholder(tf.float32, shape=[])
		self.D_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.D_loss, var_list=self.discriminator.vars)
		self.G_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.G_loss, var_list=self.generator.vars)
		
		self.saver = tf.train.Saver()
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.model_name = 'Models/ebgan.ckpt'

	def train(self, sample_dir, training_epoches = 500000, batch_size = 32):
		fig_count = 0
		self.sess.run(tf.global_variables_initializer())
		#self.saver.restore(self.sess, self.model_name)		

		learning_rate_initial = 1e-4
		for epoch in range(training_epoches):
			learning_rate =  learning_rate_initial * pow(0.5, epoch // 50000)
			# update D and G
			X_b = self.data(batch_size)
			self.sess.run(
				[self.D_solver, self.G_solver],
				feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim), self.learning_rate: learning_rate}
				)
			# save img, model. print loss
			if epoch % 100 == 0 or epoch < 100:
				D_loss_curr, G_loss_curr = self.sess.run(
						[self.D_loss, self.G_loss],
            			feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)})
				print('Iter: {}; D loss: {:.4}; G_loss: {:.4};'.format(epoch, D_loss_curr, G_loss_curr))

				if epoch % 1000 == 0:
					X_s, real, samples = self.sess.run([self.X, self.D_real, self.G_sample], feed_dict={self.X: X_b[:16,:,:,:], self.z: sample_z(16, self.z_dim)})

					fig = self.data.data2fig(X_s)
					plt.savefig('{}/{}.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
					plt.close(fig)

					fig = self.data.data2fig(real)
					plt.savefig('{}/{}_d.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
					plt.close(fig)

					fig = self.data.data2fig(samples)
					plt.savefig('{}/{}_r.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
					plt.close(fig)

					fig_count += 1

				if epoch % 5000 == 0:
					self.saver.save(self.sess, self.model_name)

if __name__ == '__main__':

	# constraint GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	# save generated images
	sample_dir = 'Samples/ebgan'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)

	# param
	generator = G_conv()
	discriminator = D_autoencoder()

	data = celebA()

	# run
	ebgan = EBGAN(generator, discriminator, data)
	ebgan.train(sample_dir)

