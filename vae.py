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
	return np.random.uniform(0, 1., size=[m, n])

class VAE():
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

		# nets
		mu, sigma = self.discriminator(self.X) 
		latent_code = mu + tf.exp(sigma/2)*self.z
		
		self.G_real = self.generator(latent_code)
		self.G_sample = self.generator(self.z)
		
		# loss
		# E[log P(X|z)]
		epsilon = 1e-8
		self.recon = tf.reduce_sum(-self.X * tf.log(self.G_real + epsilon) -(1.0 - self.X) * tf.log(1.0 - self.G_real + epsilon))
		
		# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
		self.kl = 0.5 * tf.reduce_sum(tf.exp(sigma) + tf.square(mu) - 1. - sigma)

		self.loss = self.recon + self.kl

		# solver
		self.learning_rate = tf.placeholder(tf.float32, shape=[])
		self.solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.generator.vars + self.discriminator.vars)
		
		self.saver = tf.train.Saver()
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.model_name = 'Models/vae_cifar.ckpt'

	def train(self, sample_dir, training_epoches = 500000, batch_size = 32):
		fig_count = 0
		self.sess.run(tf.global_variables_initializer())
		#self.saver.restore(self.sess, self.model_name)		

		learning_rate_initial = 1e-4
		for epoch in range(training_epoches):
			learning_rate = learning_rate_initial * pow(0.5, epoch // 50000)
			X_b = self.data(batch_size)
			self.sess.run(
				self.solver,
				feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim), self.learning_rate: learning_rate}
				)
			# save img, model. print loss
			if epoch % 100 == 0 or epoch < 100:
				loss_curr = self.sess.run(
						self.loss,
            			feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)})
				print('Iter: {}; loss: {:.4}'.format(epoch, loss_curr))

				if epoch % 1000 == 0:
					real, samples = self.sess.run([self.G_real, self.G_sample], feed_dict={self.X: X_b[:16,:,:,:], self.z: sample_z(16, self.z_dim)})

					fig = self.data.data2fig(real)
					plt.savefig('{}/{}.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
					plt.close(fig)

					fig = self.data.data2fig(samples)
					plt.savefig('{}/{}_s.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
					plt.close(fig)

					fig_count += 1

				if epoch % 5000 == 0:
					self.saver.save(self.sess, self.model_name)


if __name__ == '__main__':

	# constraint GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = '2'

	# save generated images
	sample_dir = 'Samples/vae'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)

	# param
	generator = G_conv()
	discriminator = D_vae()

	data = celebA()

	# run
	vae = VAE(generator, discriminator, data)
	vae.train(sample_dir)

