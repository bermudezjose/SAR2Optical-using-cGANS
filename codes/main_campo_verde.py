import argparse
import os
import scipy.misc
import numpy as np

# from model import pix2pix
# from model_128 import pix2pix
# from model_semiautomatic import pix2pix
# from model_semi_campo_verde import pix2pix
# from model_semi_campo_verde_unsupervised import pix2pix
# from model_campo_verde_sythetize_semisupervised import pix2pix
# from model_campo_verde_sythetize_unsupervised import pix2pix
# from model_semi_campo_verde_baseline import pix2pix
# from model_semiautomatic_quemadas import pix2pix
# from model_campo_verde_sythetize_semisupervised_multitemporal import pix2pix
# from model_campo_verde_sythetize_unsupervised_supervised_multitemporal_ import pix2pix
# from model_campo_verde_fcn import pix2pix
# from model_campo_verde_semantic_discriminator import pix2pix
from model_campo_verde_semantic_discriminator_martha import pix2pix
# from model_campo_verde_fcn_semi import pix2pix
# from model_campo_verde_fcn_semi_offline import pix2pix
# from model_campo_verde_fcn_semi_backup2 import pix2pix

import tensorflow as tf
parser = argparse.ArgumentParser(description='')
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
#parser.add_argument('--dataset_name', dest='dataset_name', default='11nov2015_C01', help='name of the dataset')
#parser.add_argument('--experiment_type', dest='experiment_type', default='case_01', help='define experiment case')
#parser.add_argument('--image_region', dest='image_region', default='top', help='define region de la imagen a analizar')
#parser.add_argument('--dataset_name', dest='dataset_name', default='27nov2015_C01_crops', help='name of the dataset')
#parser.add_argument('--experiment_type', dest='experiment_type', default='case_01', help='define experiment case')
#parser.add_argument('--image_region', dest='image_region', default='all', help='define region de la imagen a analizar')
#parser.add_argument('--dataset_name', dest='dataset_name', default='13dec2015_C01', help='name of the dataset')
#parser.add_argument('--experiment_type', dest='experiment_type', default='case_01', help='define experiment case')
#parser.add_argument('--image_region', dest='image_region', default='all', help='define region de la imagen a analizar')
#parser.add_argument('--dataset_name', dest='dataset_name', default='18mar2016_C01', help='name of the dataset')
#parser.add_argument('--experiment_type', dest='experiment_type', default='case_01', help='define experiment case')
#parser.add_argument('--image_region', dest='image_region', default='top', help='define region de la imagen a analizar')
#parser.add_argument('--dataset_name', dest='dataset_name', default='05may2016_C01', help='name of the dataset')
#parser.add_argument('--experiment_type', dest='experiment_type', default='case_01', help='define experiment case')
#parser.add_argument('--image_region', dest='image_region', default='top', help='define region de la imagen a analizar')
#parser.add_argument('--dataset_name', dest='dataset_name', default='08jul2016_C01_T', help='name of the dataset')
#parser.add_argument('--experiment_type', dest='experiment_type', default='case_01', help='define experiment case')
#parser.add_argument('--image_region', dest='image_region', default='top', help='define region de la imagen a analizar')
#parser.add_argument('--dataset_name', dest='dataset_name', default='24jul2016_C01', help='name of the dataset')
#parser.add_argument('--experiment_type', dest='experiment_type', default='case_01', help='define experiment case')
#parser.add_argument('--image_region', dest='image_region', default='top', help='define region de la imagen a analizar')

# parser.add_argument('--dataset_name', dest='dataset_name', default='05may2016_C01_semi_baseline', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='05may2016_C01_synthesize_semisupervised_baseline', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='05may2016_C01_synthesize_unsuper_super', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='05may2016_C01_semantic_discriminator', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

parser.add_argument('--dataset_name', dest='dataset_name', default='05may2016_C01_semantic_discriminator_martha_L2', help='name of the dataset')
parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='05may2016_C01_synthesize_fcn_BASELINE_FINETUNING', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='05may2016_C01_synthesize_fcn_BASELINE', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')


# parser.add_argument('--dataset_name', dest='dataset_name', default='05may2016_C01_synthesize_semisupervised_wholeimageL1_10', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='05may2016_C01_semi_dropout', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='05may2016_C01', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='quemadas_ap2_case_C', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_C', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='quemadas_ap2_case_C_new', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_C', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='quemadas_ap2_case_A', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')
 
# parser.add_argument('--dataset_name', dest='dataset_name', default='quemadas_ap2_case_A_semi_L0', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='quemadas_ap2_case_A_semi_L02', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='quemadas_ap2_case_A_semi_L03_all_labeled', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='quemadas_ap2_case_A_valset', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='quemadas_ap2_case_A_semi', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='quemadas_ap2_case_A_nosemi', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')

# parser.add_argument('--dataset_name', dest='dataset_name', default='quemadas_ap2_case_A_new', help='name of the dataset')
# parser.add_argument('--experiment_type', dest='experiment_type', default='case_A', help='define experiment case')


parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=4000, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=11, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=7, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test, generate_image, create_dataset')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--isTrain', dest='isTrain', type=bool, default=True, help='enable traninable parameters')

args = parser.parse_args()
name = 'config_' + args.dataset_name
# np.save(name, args)


def main(_):
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # config = tf.ConfigProto(device_count = {'GPU': 1})
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.Session(config=config) as sess:
        model = pix2pix(sess, image_size=args.fine_size, load_size = args.load_size, batch_size=args.batch_size,
                        output_size=args.fine_size, input_c_dim=args.input_nc, L1_lambda=args.L1_lambda,
                        output_c_dim=args.output_nc, dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir, isTrain=args.isTrain)

        if args.phase == 'train':
            model.train(args)
        elif args.phase == 'fine_tuning':
            model.fine_tuning(args)
        elif args.phase == 'test':
            model.test(args)
        elif args.phase == 'generate_image':
            model.generate_image(args)
        elif args.phase == 'create_dataset':
            model.create_dataset(args)
        elif args.phase == 'classify_image':
            model.classiffy_generated_image(args)
        elif args.phase == 'generate_features':
            model.generate_features(args)
        elif args.phase == 'compare':
            model.compare_rel_vs_fake_trnsamples(args) 
        else:
            print ('Invalid option ...')

if __name__ == '__main__':
    tf.app.run()
