import os
import argparse import ArgumentParser
from datasets import *



def build_parser():
	parser = ArgumentParser()
	parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
	parser.add_argument('--task_name', type=str, default='facescrub', help='Set data name')
	parser.add_argument('--epoch_size', type=int, default=5000, help='Set epoch size')
	parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
	parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
	parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
	parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
	parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
	parser.add_argument('--image_size', type=int, default=64, help='Image size. 64 for every experiment in the paper')

	parser.add_argument('--gan_curriculum', type=int, default=10000, help='Strong GAN loss for certain period at the beginning')
	parser.add_argument('--starting_rate', type=float, default=0.01, help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
	parser.add_argument('--default_rate', type=float, default=0.5, help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')

	parser.add_argument('--style_A', type=str, default=None, help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
	parser.add_argument('--style_B', type=str, default=None, help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
	parser.add_argument('--constraint', type=str, default=None, help='Constraint for celebA dataset. Only images satisfying this constraint is used. For example, if --constraint=Male, and --constraint_type=1, only male images are used for both style/domain.')
	parser.add_argument('--constraint_type', type=str, default=None, help='Used along with --constraint. If --constraint_type=1, only images satisfying the constraint are used. If --constraint_type=-1, only images not satisfying the constraint are used.')
	parser.add_argument('--n_test', type=int, default=200, help='Number of test data.')

	parser.add_argument('--update_interval', type=int, default=3, help='')
	parser.add_argument('--log_interval', type=int, default=50, help='Print loss values every log_interval iterations.')
	parser.add_argument('--image_save_interval', type=int, default=1000, help='Save test results every image_save_interval iterations.')
	parser.add_argument('--model_save_interval', type=int, default=10000, help='Save models every model_save_interval iterations.')

	return parser

def as_np(data):
    return data.cpu().data.numpy()

def get_data():
    # celebA / edges2shoes / edges2handbags / ...
    if args.task_name == 'facescrub':
        data_A, data_B = get_facescrub_files(test=False, n_test=args.n_test)
        test_A, test_B = get_facescrub_files(test=True, n_test=args.n_test)

    elif args.task_name == 'celebA':
        data_A, data_B = get_celebA_files(style_A=args.style_A, style_B=args.style_B, constraint=args.constraint, constraint_type=args.constraint_type, test=False, n_test=args.n_test)
        test_A, test_B = get_celebA_files(style_A=args.style_A, style_B=args.style_B, constraint=args.constraint, constraint_type=args.constraint_type, test=True, n_test=args.n_test)

    elif args.task_name == 'edges2shoes':
        data_A, data_B = get_edge2photo_files( item='edges2shoes', test=False )
        test_A, test_B = get_edge2photo_files( item='edges2shoes', test=True )

    elif args.task_name == 'edges2handbags':
        data_A, data_B = get_edge2photo_files( item='edges2handbags', test=False )
        test_A, test_B = get_edge2photo_files( item='edges2handbags', test=True )

    elif args.task_name == 'handbags2shoes':
        data_A_1, data_A_2 = get_edge2photo_files( item='edges2handbags', test=False )
        test_A_1, test_A_2 = get_edge2photo_files( item='edges2handbags', test=True )

        data_A = np.hstack( [data_A_1, data_A_2] )
        test_A = np.hstack( [test_A_1, test_A_2] )

        data_B_1, data_B_2 = get_edge2photo_files( item='edges2shoes', test=False )
        test_B_1, test_B_2 = get_edge2photo_files( item='edges2shoes', test=True )

        data_B = np.hstack( [data_B_1, data_B_2] )
        test_B = np.hstack( [test_B_1, test_B_2] )

    return data_A, data_B, test_A, test_B

# get_fm_loss()
def get_fm_loss(real_feats, fake_feats):
	loss = 0
	losses = []
    for real_feat, fake_feat in zip(real_feats, fake_feats):
		l2 = (tf.reduce_mean(real_feat,0)-tf.reduce_mean(fake_feat,0))
			* (tf.reduce_mean(real_feat,0)-tf.reduce_mean(fake_feat,0))
		losses.append(tf.reduce_mean(l2))
	loss += reduce(tf.add, losses)

	return loss

# get_gan_loss()
def get_gan_loss(dis_real, dis_fake):
	labels_dis_real = tf.constant(1,shape=[tf.shape(dis_real)[0],1])
	labels_dis_fake = tf.constant(0,shape=[tf.shape(dis_fake)[0],1])
	labels_gen = tf.constant(1, shape=[tf.shape(dis_fake)[0],1])

	dis_loss = tf.sigmoid_cross_entropy_with_logits(logits=dis_real,labels=labels_dis_real)*0.5
		+ tf.sigmoid_cross_entropy_with_logits(logits=dis_fake,labels=labels_dis_fake)*0.5
	dis_loss = tf.sigmoid_cross_entropy_with_logits(logits=dis_fake,labels=labels_gen)
return dis_loss, gen_loss

def main():
	global args
	parser = build_parser()
	args = parser.parse_args()

	task_name = args.task_name

    epoch_size = args.epoch_size
    batch_size = args.batch_size

	WEIGHT_DECAY = 0.00001

    result_path = os.path.join( args.result_path, args.task_name )
    if args.style_A:
        result_path = os.path.join( result_path, args.style_A )
    result_path = os.path.join( result_path, args.model_arch )

    model_path = os.path.join( args.model_path, args.task_name )
    if args.style_A:
        model_path = os.path.join( model_path, args.style_A )
    model_path = os.path.join( model_path, args.model_arch )

    data_style_A, data_style_B, test_style_A, test_style_B = get_data()

    if args.task_name.startswith('edges2'):
        test_A = read_images( test_style_A, 'A', args.image_size )
        test_B = read_images( test_style_B, 'B', args.image_size )

    elif args.task_name == 'handbags2shoes' or args.task_name == 'shoes2handbags':
        test_A = read_images( test_style_A, 'B', args.image_size )
        test_B = read_images( test_style_B, 'B', args.image_size )

    else:
        test_A = read_images( test_style_A, None, args.image_size )
        test_B = read_images( test_style_B, None, args.image_size )

	test_A = tf.placeholder("float")
	test_B = tf.placeholder("float")

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

	# generator_A = Generator(A) # returns a tensor
	# generator_B = Generator(B)
    # discriminator_A = Discriminator(A)
    # discriminator_B = Discriminator(B)

    # if cuda:
    #     test_A = test_A.cuda()
    #     test_B = test_B.cuda()
    #     generator_A = generator_A.cuda()
    #     generator_B = generator_B.cuda()
    #     discriminator_A = discriminator_A.cuda()
    #     discriminator_B = discriminator_B.cuda()

	data_size = min( len(data_style_A), len(data_style_B) )
    n_batches = ( data_size // batch_size ) # //: floor operator

    # recon_criterion = nn.MSELoss() # Mean Squared Error
    # gan_criterion = nn.BCELoss() # Binary Cross Entropy
    # feat_criterion = nn.HingeEmbeddingLoss() # to measure if 2 inputs are similar

    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

	batch_index = 1

 	# set up decay rate as 0.00001
	learning_rate = tf.train.exponentional_decay(learning_rate=args.learning_rate,
												global_step=1,
												decay_steps=1,
												decay_rate=0.99999)
	optim_gen = tf.train.AdamOptimizer(learning_rate, 0.5)
	optim_dis = tf.train.AdamOptimizer(learning_rate, 0.5)

    iters = 0

    gen_loss_total = []
    dis_loss_total = []

	init = tr.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	for i in range(1, epoch_size+1):
    # for epoch in range(epoch_size):
        data_style_A, data_style_B = shuffle_data( data_style_A, data_style_B)

        widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval=n_batches, widgets=widgets)
        pbar.start()

        for i in range(n_batches):

            pbar.update(i)

            # generator_A.zero_grad() TODO: do we need to set all gradient in model to 0 in tf?
            # generator_B.zero_grad()
            # discriminator_A.zero_grad()
            # discriminator_B.zero_grad()

            A_path = data_style_A[ i * batch_size: (i+1) * batch_size ]
            B_path = data_style_B[ i * batch_size: (i+1) * batch_size ]

            if args.task_name.startswith( 'edges2' ):
                A = read_images( A_path, 'A', args.image_size ) # upper half of input collection, sketch
                B = read_images( B_path, 'B', args.image_size ) # lower half of input collection, colored picture
            elif args.task_name =='handbags2shoes' or args.task_name == 'shoes2handbags':
                A = read_images( A_path, 'B', args.image_size )
                B = read_images( B_path, 'B', args.image_size )
            else:
                A = read_images( A_path, None, args.image_size )
                B = read_images( B_path, None, args.image_size )

			A = tf.convert_to_tensor(A)
			B = tf.convert_to_tensor(B)

			AB = Generator(A)
			BA = Generator(B)

			ABA = Generator(AB)
			BAB = Generator(BA)

			# Reconstruction Loss
			recon_loss_A = tf.metrics.mean_squared_error(A, ABA)
			recon_loss_B = tf.metrics.mean_squared_error(B, BAB)

			# Real/Fake GAN Loss (A)
            A_dis_real, A_feats_real = discriminator_A( A )
            A_dis_fake, A_feats_fake = discriminator_A( BA )

			dis_loss_B, gen_loss_B = get_gan_loss( B_dis_real, B_dis_fake, gan_criterion, cuda )
            fm_loss_B = get_fm_loss( B_feats_real, B_feats_fake, feat_criterion )

			# Total Loss
			if args.model_arch == 'discogan':
                gen_loss = gen_loss_A_total + gen_loss_B_total
                dis_loss = dis_loss_A + dis_loss_B
            elif args.model_arch == 'recongan':
                gen_loss = gen_loss_A_total
                dis_loss = dis_loss_B
            elif args.model_arch == 'gan':
                gen_loss = (gen_loss_B*0.1 + fm_loss_B*0.9)
                dis_loss = dis_loss_B

			# (backward gradient, update argument in pytorch)

			if iters % args.log_interval == 0:
				print "---------------------"
                print "GEN Loss:", as_np(gen_loss_A.mean()), as_np(gen_loss_B.mean())
                print "Feature Matching Loss:", as_np(fm_loss_A.mean()), as_np(fm_loss_B.mean())
                print "RECON Loss:", as_np(recon_loss_A.mean()), as_np(recon_loss_B.mean())
                print "DIS Loss:", as_np(dis_loss_A.mean()), as_np(dis_loss_B.mean())

			if iters % args.image_save_interval == 0:
				AB = Generator(test_A)
				BA = Generator(test_B)
				ABA = Generator(AB)
				BAB = Generator(BA)

				n_testset = min( test_A.size()[0], test_B.size()[0] )
                subdir_path = os.path.join( result_path, str(iters / args.image_save_interval) )

                if os.path.exists( subdir_path ):
                    pass
                else:
                    os.makedirs( subdir_path )

				for im_idx in range( n_testset ):
