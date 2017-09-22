import tensorflow as tf

# ============== main starts here ==============================
# define some arguments
WEIGHT_DECAY = 0.00001

global args
parser = build_parser()
args = parser.parse_args()

task_name = args.task_name
epoch_size = args.epoch_size
batch_size = args.batch_size

# define path name
result_path = os.path.join( args.result_path, args.task_name )
if args.style_A:
    result_path = os.path.join( result_path, args.style_A )
result_path = os.path.join( result_path, args.model_arch )

model_path = os.path.join( args.model_path, args.task_name )
if args.style_A:
    model_path = os.path.join( model_path, args.style_A )
model_path = os.path.join( model_path, args.model_arch )

if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# get data
data_style_A, data_style_B, test_style_A, test_style_B = get_data()
# assume task_name can only start with 'edges2'
test_A = read_images( test_style_A, 'A', args.image_size )
test_B = read_images( test_style_B, 'B', args.image_size )

test_A = tf.convert_to_tensor("float")
test_B = tf.convert_to_tensor("float")

data_size = min( len(data_style_A), len(data_style_B) )
n_batches = ( data_size // batch_size ) # //: floor operator


# define variables for training
gen_loss_total = []
dis_loss_total = []

A = tf.placeholder(tf.float32)
B = tf.placeholder(tf.float32)
AB = Generator(A)
BA = Generator(B)
ABA = Generator(AB)
BAB = Generator(BA)

# define variables for testing
AB_saving_op = Generator(test_A)
BA_saving_op = Generator(test_B)
ABA_saving_op = Generator(AB_tosave)
BAB_saving_op = Generator(BA_tosave)

# Reconstruction Loss
recon_loss_A = tf.reduce_mean(tf.metrics.mean_squared_error(A, ABA))
recon_loss_B = tf.reduce_mean(tf.metrics.mean_squared_error(B, BAB))

# Real/Fake GAN Loss (A)
A_dis_real, A_feats_real = Discriminator( A )
A_dis_fake, A_feats_fake = Discriminator( BA )

dis_loss_A, gen_loss_A = get_gan_loss( A_dis_real, A_dis_fake)
fm_loss_A = get_fm_loss( A_feats_real, A_feats_fake)

# Real/Fake GAN Loss (B)
B_dis_real, B_feats_real = Discriminator( B )
B_dis_fake, B_feats_fake = Discriminator( AB )

dis_loss_B, gen_loss_B = get_gan_loss( B_dis_real, B_dis_fake)
fm_loss_B = get_fm_loss( B_feats_real, B_feats_fake)

if iters < args.gan_curriculum:
    rate = args.starting_rate
else:
    rate = args.default_rate

# intermediate loss functions
gen_loss_A_total = (gen_loss_B * 0.1 + fm_loss_B * 0.9) * (1. - rate) + recon_loss_A * rate
gen_loss_B_total = (gen_loss_A * 0.1 + fm_loss_A * 0.9) * (1. - rate) + recon_loss_B * rate

if args.model_arch == 'discogan':
	gen_loss = gen_loss_A_total + gen_loss_B_total
	dis_loss = dis_loss_A + dis_loss_B
elif args.model_arch == 'recongan':
	gen_loss = gen_loss_A_total
	dis_loss = dis_loss_B
elif args.model_arch == 'gan':
	gen_loss = (gen_loss_B*0.1 + fm_loss_B*0.9)
	dis_loss = dis_loss_B


# set up decay rate as 0.00001
learning_rate = tf.train.exponentional_decay(learning_rate=args.learning_rate,
											global_step=1,
											decay_steps=1,
											decay_rate=0.99999)
optim_gen = tf.train.AdamOptimizer(learning_rate, 0.5).minimize(gen_loss)
optim_dis = tf.train.AdamOptimizer(learning_rate, 0.5).minimize(dis_loss)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_index = 1
iters = 0

for i in range(1, epoch_size+1):
	# reading batch data
    data_style_A, data_style_B = shuffle_data( data_style_A, data_style_B)

    widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval=n_batches, widgets=widgets)
    pbar.start()

    for i in range(n_batches):
        pbar.update(i)

        A_path = data_style_A[ i * batch_size: (i+1) * batch_size ]
        B_path = data_style_B[ i * batch_size: (i+1) * batch_size ]

        A = read_images( A_path, 'A', args.image_size ) # upper half of input collection, sketch
        B = read_images( B_path, 'B', args.image_size ) # lower half of input collection, colored picture

		if iters % args.log_interval == 0:
			print "---------------------"
			print "GEN Loss", as_np(tf.reduce_mean(gen_loss_A)), as_np(tf.reduce_mean(gen_loss_B))
			print "Feature Matching Loss:", as_np(tf.reduce_mean(fm_loss_A)), as_np(tf.reduce_mean(fm_loss_B))
			print "RECON Loss:", as_np(tf.reduce_mean(recon_loss_A)), as_np(tf.reduce_mean(recon_loss_B))
			print "DIS Loss:", as_np(tf.reduce_mean(dis_loss_A)), as_np(tf.reduce_mean(dis_loss_B))

		if iters % args.image_save_interval == 0:
			AB,BA,ABA,BAB=sess.run([AB_saving_op,BA_saving_op,ABA_saving_op,BAB_saving_op])

			n_testset = min( test_A.size()[0], test_B.size()[0] )
            subdir_path = os.path.join( result_path, str(iters / args.image_save_interval) )

            if os.path.exists( subdir_path ):
                pass
            else:
                os.makedirs( subdir_path )

			for im_idx in range( n_testset ):
				A_val = sess.run(test_A[im_idx]).transpose(1,2,0) * 255.
				B_val = sess.run(test_B[im_idx]).transpose(1,2,0) * 255.
				BA_val = sess.run(BA[im_idx]).transpose(1,2,0)*255.
				ABA_val = sess.run(ABA[im_idx]).transpose(1,2,0)*255.
				AB_val = sess.run(AB[im_idx]).transpose(1,2,0)*255.
				BAB_val = sess.run(BAB[im_idx]).transpose(1,2,0)*255.

                filename_prefix = os.path.join (subdir_path, str(im_idx))
                scipy.misc.imsave( filename_prefix + '.A.jpg', A_val.astype(np.uint8)[:,:,::-1])
                scipy.misc.imsave( filename_prefix + '.B.jpg', B_val.astype(np.uint8)[:,:,::-1])
                scipy.misc.imsave( filename_prefix + '.BA.jpg', BA_val.astype(np.uint8)[:,:,::-1])
                scipy.misc.imsave( filename_prefix + '.AB.jpg', AB_val.astype(np.uint8)[:,:,::-1])
                scipy.misc.imsave( filename_prefix + '.ABA.jpg', ABA_val.astype(np.uint8)[:,:,::-1])
                scipy.misc.imsave( filename_prefix + '.BAB.jpg', BAB_val.astype(np.uint8)[:,:,::-1])

		if iters % args.model_save_interval == 0:
			save_path = saver.save(sess, os.path.join(model_path, str(iters/args.model_save_interval)))

		iters += 1
