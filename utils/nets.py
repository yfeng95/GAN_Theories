import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

class G_conv(object):
	def __init__(self, channel=3, name='G_conv'):
		self.name = name
		self.size = 64/16
		self.channel = channel

	def __call__(self, z):
		with tf.variable_scope(self.name) as scope:
			g = tcl.fully_connected(z, self.size * self.size * 512, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			g = tf.reshape(g, (-1, self.size, self.size, 512))  # size
			g = tcl.conv2d_transpose(g, 256, 3, stride=2, # size*2
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.conv2d_transpose(g, 128, 3, stride=2, # size*4
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.conv2d_transpose(g, 64, 3, stride=2, # size*8 32x32x64
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			
			g = tcl.conv2d_transpose(g, self.channel, 3, stride=2, # size*16 
										activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			return g
	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class D_conv(object):
	def __init__(self, name='D_conv'):
		self.name = name

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			size = 64
			d = tcl.conv2d(x, num_outputs=size, kernel_size=3, # bzx64x64x3 -> bzx32x32x64
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.conv2d(d, num_outputs=size * 2, kernel_size=3, # 16x16x128
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.conv2d(d, num_outputs=size * 4, kernel_size=3, # 8x8x256
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.conv2d(d, num_outputs=size * 8, kernel_size=3, # 4x4x512
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

			d = tcl.fully_connected(tcl.flatten(d), 256, activation_fn=lrelu, weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.fully_connected(d, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
			
			return d
			
	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

# for ebgan and began
class D_autoencoder(object):
	def __init__(self, n_hidden=256, name='D_autoencoder'):
		self.name = name
		self.n_hidden = n_hidden

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			# --- conv
			size = 64
			d = tcl.conv2d(x, num_outputs=size, kernel_size=3, # bzx64x64x3 -> bzx32x32x64
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.conv2d(d, num_outputs=size * 2, kernel_size=3, # 16x16x128
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.conv2d(d, num_outputs=size * 4, kernel_size=3, # 8x8x256
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.conv2d(d, num_outputs=size * 8, kernel_size=3, # 4x4x512
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
	
			h = tcl.fully_connected(tcl.flatten(d), self.n_hidden, activation_fn=lrelu, weights_initializer=tf.random_normal_initializer(0, 0.02))

			# -- deconv
			d = tcl.fully_connected(h, 4 * 4 * 512, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			d = tf.reshape(d, (-1, 4, 4, 512))  # size
			d = tcl.conv2d_transpose(d, 256, 3, stride=2, # size*2
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.conv2d_transpose(d, 128, 3, stride=2, # size*4
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.conv2d_transpose(d, 64, 3, stride=2, # size*8
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			
			d = tcl.conv2d_transpose(d, 3, 3, stride=2, # size*16
									activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			return d

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

# for vae
class D_vae(object):
	def __init__(self, name='D_vae'):
		self.name = name

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			size = 64
			d = tcl.conv2d(x, num_outputs=size, kernel_size=3, # bzx64x64x3 -> bzx32x32x64
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.conv2d(d, num_outputs=size * 2, kernel_size=3, # 16x16x128
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.conv2d(d, num_outputs=size * 4, kernel_size=3, # 8x8x256
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.conv2d(d, num_outputs=size * 8, kernel_size=3, # 4x4x512
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

			d = tcl.fully_connected(tcl.flatten(d), 256, activation_fn=lrelu, weights_initializer=tf.random_normal_initializer(0, 0.02))
			mu = tcl.fully_connected(d, 100, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
			sigma = tcl.fully_connected(d, 100, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
			
			return mu, sigma
			
	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)



