import torch
from visualization import *
from torchvision.transforms import transforms
from model import GetG_net
from datasets import Get_Data_Test, Get_Data_Index
import numpy as np
import torch.nn as nn


def jaccard(y_true, y_pred, args):
    y_true = y_true.detach().cpu().view(-1, args.image_channels, args.image_H, args.image_W)
    y_true = transforms.Grayscale()(y_true)
    y_true = (y_true.numpy() > 0.5).astype(np.uint8)

    y_pred = y_pred.detach().cpu().view(-1, args.image_channels, args.image_H, args.image_W)
    y_pred = transforms.Grayscale()(y_pred)
    y_pred = (y_pred.numpy() > 0.5).astype(np.uint8)

    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred, args):
    y_true = y_true.detach().cpu().view(-1, args.image_channels, args.image_H, args.image_W)
    y_true = transforms.Grayscale()(y_true)
    y_true = (y_true.numpy() > 0.5).astype(np.uint8)

    y_pred = y_pred.detach().cpu().view(-1, args.image_channels, args.image_H, args.image_W)
    y_pred = transforms.Grayscale()(y_pred)
    y_pred = (y_pred.numpy() > 0.5).astype(np.uint8)

    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Train parameters
args = TrainParameters().parse()

# Model initialization generator and discriminator
# mode means different method(0:I2IS-1D, 1:I2IS-1cD, 2: I2IS-2D, 3:dualGAN, 4:SL, 5:pix2pix)
if args.method == '/I2IS-1D/':
    method = 0
if args.method == '/I2IS-1cD/':
    method = 1
if args.method == '/I2IS-2D/':
    method = 2
if args.method == '/dualGAN/':
    method = 3
if args.method == '/SL/':
    method = 4
if args.method == '/pix2pix/':
    method = 5
    
G = GetG_net(device, method)

##################################################visualization of test###################################################
# Configure DataLoaders for visualization
test_loader = Get_Data_Test(args)

img_list = []

# visualization for one batch_size_test !!! change its value
X, labels = next(iter(test_loader))  # Get a single batch from DataLoader without iterating

# from left to right
image_size = X.size(3) // 5
img_A = X[:, :, :, :image_size].to(device)  # input image
img_B = X[:, :, :, image_size:2 * image_size].to(device)  # disordered mask
img_C = X[:, :, :, 2 * image_size:3 * image_size].to(device)  # disordered label
img_D = X[:, :, :, 3 * image_size:4 * image_size].to(device)  # true mask
img_E = X[:, :, :, 4 * image_size:].to(device)  # true label

real_inputs = torch.cat([img_A, img_E], dim=1)

# fake data
G_output = G(img_A)

G_output = torch.mul(img_A, G_output)  # new loss
X_fake = torch.cat([real_inputs, G_output], dim=1)

img_list = To_Image(img_list, X_fake)  # store all figures in img_list
Show_GeneratedImages(img_list)

##################################################index###############################################################
# Configure DataLoaders for index
test_loader_index = Get_Data_Index(args)

pixAccurate = 0
result_jaccard = 0
result_dice = 0
times = 0
for times, (X, labels) in enumerate(test_loader_index):
    times += 1

    # from left to right
    image_size = X.size(3) // 5
    img_A = X[:, :, :, :image_size].to(device)  # input image
    img_B = X[:, :, :, image_size:2*image_size].to(device)  # disordered mask
    img_C = X[:, :, :, 2*image_size:3*image_size].to(device)  # disordered label
    img_D = X[:, :, :, 3*image_size:4*image_size].to(device)  # true mask
    img_E = X[:, :, :, 4*image_size:].to(device)  # true label

    real_inputs = torch.cat([img_A, img_E], dim=1)

    G_output = G(img_A)
    G_output1 = torch.mul(img_A, G_output)  # new loss
        
    # compute average pixel accurate pixAccurate
    # pixAccurate += nn.L1Loss()(torch.mul(G_output, img_D), img_E)
    pixAccurate += nn.L1Loss()(G_output1, img_E)
        
    result_jaccard += jaccard(img_D, G_output, args)
    result_dice += dice(img_D, G_output, args)

print('pixAccurate = ', 10*pixAccurate/times)
print('jaccard = ', result_jaccard/times)
print('dice = ', result_dice/times)
