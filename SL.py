import time
from datasets import *
from model import *
from optimizer import *
from visualization import *

# build a file in the current path. For the generated image, losses and model parameter saving
img_save_path1 = "images"
os.makedirs(img_save_path1, exist_ok=True) 
img_save_path2 = "losses"
os.makedirs(img_save_path2, exist_ok=True)
img_save_path3 = "model"
os.makedirs(img_save_path3, exist_ok=True)

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Train parameters
args = TrainParameters().parse()

# Model initialization generator and discriminator.
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
G, G2, D1, D2 = Create_nets(device, method)

# Optimizers
G_optimizer, G2_optimizer, D1_optimizer, D2_optimizer = Get_optimizers(args, G, G2, D1, D2)

# Configure DataLoaders
train_loader = Get_Data_Train(args)
# Show_ImageDataSet(device, train_loader)

# Train
# Lists to keep track of progress
start_time = time.time()  # for the computation of training time
G_losses = []  # save g_loss
D_losses = []  # save d_loss
g_loss = []
d_loss = []
fake_inputs = []
real_inputs = []
img_list = []  # store generated images

# start training
for epoch in range(args.epochs):
    epoch += 1

    for times, (X, labels) in enumerate(train_loader):
        times += 1

        # from left to right
        image_size = X.size(3) // 5
        img_A = X[:, :, :, :image_size].to(device)  # input image
        img_B = X[:, :, :, image_size:2*image_size].to(device)  # disordered mask
        img_C = X[:, :, :, 2*image_size:3*image_size].to(device)  # disordered label
        img_D = X[:, :, :, 3*image_size:4*image_size].to(device)  # true mask
        img_E = X[:, :, :, 4*image_size:].to(device)  # true label

        real_inputs = torch.cat([img_A, img_E], dim=1)

        '''
        # ------------------
        #  Train Generators
        # ------------------
        '''
        # for fake data
        fake_inputs = G(img_A)

        # Backward propagation
        G_L1_Loss = nn.L1Loss()(fake_inputs, img_D).to(device)
        g_loss = G_L1_Loss

        G_optimizer.zero_grad()  # Zero the parameter gradients
        g_loss.backward()
        G_optimizer.step()  # Update D's parameters

        if epoch % 1 == 0:
            if times % 51 == 0 or times == len(train_loader):
                print('[{}/{}, {}/{}]  G_loss: {:.4f}'.format(epoch, args.epochs, times, len(train_loader),
                                                              g_loss.item()))

    ############################
    # for visualization
    ###########################
    # # store generated images. Check how the generator is doing by saving G's output on fixed_noise
    # if epoch == 4 or epoch == 8 or epoch == 16 or epoch == 32 or epoch == 64 or epoch == 128 or epoch == 200:
    if epoch % args.image_show_interval == 0:
        
        with torch.no_grad():
            fake_M = G(img_A)
            fake_L_inputs = torch.mul(img_A, fake_M) # new loss
            X_fake = torch.cat([img_A, fake_L_inputs], dim=1)
        img_list = To_Image(img_list, X_fake)  # store all figures in img_list

        # # Show generated images. Note: image_show_interval should be no smaller than image_store_interval
        #Show_GeneratedImages(img_list)

    # Save Losses for plotting later
    G_losses.append(g_loss.item())
    D_losses.append(g_loss.item())  # the same with g_loss

    # if epoch == args.epochs:
    if epoch % args.mode_save_interval == 0:
        # # Show final image comparision
        # Show_ImagesComparision(device, train_loader, img_list)
        # # save network during the training
        torch.save(G.state_dict(), args.model_root + args.method + args.case_train + '/G.pth')
        print('Model saved')
        
    if epoch == args.epochs:
        # # after training, write the losses to a .txt file
        LossVsEpoch(G_losses, D_losses)  # Loss versus training iteration
        LossWrite(G_losses, D_losses)

print('Training Finished.')
print('Cost Time: {}s'.format(time.time() - start_time))


Save_TrainingVideo(args.image_store_interval, img_list)