import argparse


class TrainParameters:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # Set data set root
        self.parser.add_argument('--train_root', type=str,
                                 default="/content/drive/My Drive/code/facades/train",
                                 help='dir of the train dataset')
        self.parser.add_argument('--test_root', type=str,
                                 default="/content/drive/My Drive/code/facades/test",
                                 help='dir of the test dataset')
        self.parser.add_argument('--model_root', type=str,
                                 default="/content/drive/My Drive/code/model",
                                 help='dir of the saved model parameter')
        self.parser.add_argument('--losses_root', type=str,
                                 default="/content/drive/My Drive/code/losses",
                                 help='dir of the saved losses')
                                 
        # select cases
        self.parser.add_argument('--case_train', type=str, default="case1", help='case selection for training') #set!!!
        self.parser.add_argument('--case_test', type=str, default="case1", help='case selection for test') #set!!!
                                 
        # the method used for training
        # mode means different method(0:I2IS-1D, 1:I2IS-1cD, 2: I2IS-2D, 3:dualGAN, 4:SL, 5:pix2pix)
        self.parser.add_argument('--method', type=str, default="/I2IS-2D/", help='training method') #set!!!
        # If the pretrained model will be used: 0: retrain         1: use pretrained and keep training
        self.parser.add_argument('--premodel', type=str, default= 0, help='training method') #set!!!
                                 
        # self.parser.add_argument('--root', type=str,
        #                          default="/content/drive/My Drive/code/",
        #                          help='dir of the root')

        # self.parser.add_argument('--data_root', type=str,
        #                          default="G:\\ICUBE artical and code\\code-2D1Dgreat\\code\\facades",
        #                          help='dir of the dataset')

        # Set training parameters
        self.parser.add_argument('--epochs', type=int, default=300, help='number of epochs of training')  # set!!!
        self.parser.add_argument('--batch_size', type=int, default=16, help='size of the batches for training')
        self.parser.add_argument('--batch_size_test', type=int, default=1, help='size of the batches for test')
        self.parser.add_argument('--batch_size_index', type=int, default=1, help='size of the batches for test')
        self.parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')
        self.parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')

        # Set image size
        self.parser.add_argument('--image_channels', type=int, default=3,
                                 help='number of image channels. For RGB, it is 3') 
        self.parser.add_argument('--image_H', type=int, default=256,
                                 help='the height in pixel of the trained and generated image')
        self.parser.add_argument('--image_W', type=int, default=256,
                                 help='the weight in pixel of the trained and generated image')

        # Set size of feature for conv2d
        self.parser.add_argument('--ngf', type=int, default=64,
                                 help='the depth of feature maps carried through the generator') 
        self.parser.add_argument('--ndf', type=int, default=64,
                                 help='the depth of feature maps propagated through the discriminator') 

        # For optimization
        self.parser.add_argument('--lambda_gp', type=float, default=10, help='Loss weight for gradient penalty')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='adam: decay of first order momentum of '
                                                                          'gradient')
        self.parser.add_argument('--beta2', type=float, default=0.99, help='adam: decay of second order momentum of '
                                                                            'gradient')
        self.parser.add_argument('--d_steps', type=int, default=1, help='optimization times for discriminator')
        self.parser.add_argument('--g_steps', type=int, default=1, help='optimization times for generator')
        # Can put the discriminator on higher training freq than generator

        # For visualization
        self.parser.add_argument('--image_store_interval', type=int, default=30,
                                 help='number of epochs for each image store')
        self.parser.add_argument('--image_show_interval', type=int, default=30,
                                 help='number of epochs for each image show')
        self.parser.add_argument('--mode_save_interval', type=int, default=50,
                                 help='number of epochs for each model save')  #set!!!
        self.parser.add_argument('--loss_show_interval', type=int, default=20,
                                 help='number of epochs for each loss show')
        self.parser.add_argument('--noise_dimension', type=int, default=16,
                                 help='number of generated pictures for test, '
                                      'used to define the dimension of fixed_noise. It is better to be square(int)')

    def parse(self):
        self.args = self.parser.parse_args()
        print(self.args)

        return self.args
