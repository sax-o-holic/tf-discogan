import tensorflow as tf

# Training parameters
num_steps = 70000
batch_size = 128
learning_rate = 0.0002

# Network parameters
image_dim = 784 # 28x28=784
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # noise data points

weights = {
	# 3 input, 64 output, window size 4x4
	'wc1': tf.Variable(tf.random_normal([4,4,3,64])),
	'wc2': tf.Variable(tf.random_normal([4,4,64,64*2])),
	'wc3': tf.Variable(tf.random_normal([4,4,64*2,64*4])),
	'wc4': tf.Variable(tf.random_normal([4,4,64*4,64*8])),
	'wc5': tf.Variable(tf.random_normal([4,4,64*8,100])),

	# upsampling weights
	'wd1': tf.Variable(tf.random_normal([4,4,100, 64*8])),
	'wd2': tf.Variable(tf.random_normal([4,4,64*8, 64*4])),
	'wd3': tf.Variable(tf.random_normal([4,4,64*4, 64*2])),
	'wd4': tf.Variable(tf.random_normal([4,4,64*2, 64])),
	'wd5': tf.Variable(tf.random_normal([4,4,64,3])),

	# weight for last discriminator layer
	'out': tf.Variable(tf.random_normal([4,4,64*8,1])),

}

# custom initialization (Xavier Glorot init)
def glorot_init(shape):
	return tf.random_normal(shape=shape, stddev=1./tf.sqrt(shape[0]/2.))

weights = {
	'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
	'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
	'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
	'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1]))
}
biases = {
	'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
	'gen_out': tf.Variable(tf.zeros([image_dim])),
	'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
	'disc_out': tf.Variable(tf.zeros([1]))
}

#LeakyReLU activation
def leakyrelu(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

def conv2d(x, W, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="VALID")
	return x

def conv2d_transpose(x, W, strides=1):
	x = tf.layers.conv2d_transpose(x, W, strides=[1, strides, strides, 1], padding="VALID")
	return x

def BatchNorm2d(x):
	return tf.layers.batch_normalization(x, momentum=0.1, epsilon=0.001)

def Generator(x, reuse=False):
	with tf.variable_scope('Generator', reuse=reuse):
		# conv1
		x = tf.pad(x, [[1,1],[1,1]])
		x = conv2d(x, weights['wc1'],strides=2,)
		x = leakyrelu(x)

		# conv2
		x = tf.pad(x, [[1,1],[1,1]])
		x = conv2d(x, weights['wc2'],strides=2)
		x = tf.BatchNorm2d(x)
		x = tf.LeakyReLU(x)

		# conv3
		x = tf.pad(x, [[1,1],[1,1]])
		x = conv2d(x, weights['wc3'],strides=2)
		x = tf.BatchNorm2d(x)
		x = leakyrelu(x)

		# conv4
		x = tf.pad(x, [[1,1],[1,1]])
		x = conv2d(x, weights['wc4'],strides=2)
		x = tf.BatchNorm2d(x)
		x = tf.LeakyReLU(x)

		# conv5
		x = conv2d(x, weights['wc5'],strides=1)
		x = tf.BatchNorm2d(x)
		x = tf.LeakyReLU(x)

		# transpose_conv1
		x = conv2d_transpose(x, weights['wd1'],strides=1)
		x = tf.BatchNorm2d(x)
		x = tf.nn.relu(x)

		# transpose_conv2
		x = tf.pad(x, [[1,1],[1,1]])
		x = conv2d_transpose(x, weights['wd1'],strides=2)
		x = tf.BatchNorm2d(x)
		x = tf.nn.relu(x)

		# transpose_conv3
		x = tf.pad(x, [[1,1],[1,1]])
		x = conv2d_transpose(x, weights['wd2'],strides=2)
		x = tf.BatchNorm2d(x)
		x = tf.nn.relu(x)

		# transpose_conv4
		x = tf.pad(x, [[1,1],[1,1]])
		x = conv2d_transpose(x, weights['wd3'],strides=2)
		x = tf.BatchNorm2d(x)
		x = tf.nn.relu(x)

		# transpose_conv5
		x = tf.pad(x, [[1,1],[1,1]])
		x = conv2d_transpose(x, weights['wd4'],strides=2)
		x = tf.nn.sigmoid(x)

	return x

def Discriminator(x):
	# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
	# stride=1, padding=0, dilation=1, groups=1, bias=True)
	with tf.variable_scope('Discriminator',reuse=reuse):
		# conv1
		x = tf.pac(x, [[1,1],[1,1]])
		x = conv2d(x, weights['wc1'], strides=2)
		x = leakyrelu(x)

		# conv2
		x = tf.pac(x, [[1,1],[1,1]])
		x = conv2d(x, weights['wc2'], strides=2)
		x = tf.BatchNorm2d(x)
		relu2 = leakyrelu(x)

		# conv3
		x = tf.pac(relu2, [[1,1],[1,1]])
		x = conv2d(x, weights['wc3'], strides=2)
		x = tf.BatchNorm2d(x)
		relu3 = leakyrelu(x)

		# conv4
		x = tf.pac(relu3, [[1,1],[1,1]])
		x = conv2d(x, weights['wc4'], strides=2)
		x = tf.BatchNorm2d(x)
		relu4 = leakyrelu(x)

		# out
		x = conv2d(relu4, weights['out'], strides=1)
		x = tf.nn.Sigmoid(x)

	return x, [relu2, relu3, relu4]
