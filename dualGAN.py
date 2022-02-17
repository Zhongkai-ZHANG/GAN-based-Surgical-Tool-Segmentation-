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
G_AB, G_BA, D_A, D_B = Create_nets(device, method)

# Optimizers
G_AB_optimizer, G_BA_optimizer, D_A_optimizer, D_B_optimizer = Get_optimizers(args, G_AB, G_BA, D_A, D_B)

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


for epoch in range(args.epochs):
    epoch += 1

    for times, (X, labels) in enumerate(train_loader):
        times += 1

        image_size = X.size(3) // 5
        img_A = X[:, :, :, :image_size].to(device)  # input image
        img_B = X[:, :, :, image_size:2*image_size].to(device)  # disordered mask
        img_C = X[:, :, :, 2*image_size:3*image_size].to(device)  # disordered label
        img_D = X[:, :, :, 3*image_size:4*image_size].to(device)  # true mask
        img_E = X[:, :, :, 4*image_size:].to(device)  # true label

        real_inputs = torch.cat([img_A, img_E], dim=1)
        '''      
        # ---------------------
        #  Train Discriminator
        # ---------------------
        '''
        for d_index in range(args.d_steps):

            real_A_outputs = D_A(img_A)
            real_B_outputs = D_B(img_B)
            

            fake_A_inputs = G_BA(img_B)
            fake_B_inputs = G_AB(img_A)
            
            fake_A_outputs = D_A(fake_A_inputs)
            fake_B_outputs = D_B(fake_B_inputs)

            # D loss
            # Gradient penalty
            gradient_A_penalty = compute_gradient_penalty(D_A, img_A, fake_A_inputs)
            gradient_B_penalty = compute_gradient_penalty(D_B, img_B, fake_B_inputs)
            # Adversarial loss
            D_A_loss = -torch.mean(real_A_outputs) + torch.mean(fake_A_outputs) + args.lambda_gp * gradient_A_penalty
            D_B_loss = -torch.mean(real_B_outputs) + torch.mean(fake_B_outputs) + args.lambda_gp * gradient_B_penalty
            d_loss = D_A_loss + D_B_loss

            # Backward propagation
            D_A_optimizer.zero_grad()  # Zero the parameter gradients
            D_B_optimizer.zero_grad()  # Zero the parameter gradients
        
            d_loss.backward()
        
            D_A_optimizer.step()  # Update D's parameters
            D_B_optimizer.step()  # Update D's parameters

        '''
        # ------------------
        #  Train Generators
        # ------------------
        '''
        for g_index in range(args.g_steps):

            fake_A_inputs = G_BA(img_B)
            fake_B_inputs = G_AB(img_A)
        
            fake_A_outputs = D_A(fake_A_inputs)
            fake_B_outputs = D_B(fake_B_inputs)

            # Reconstruct images
            recov_A = G_BA(fake_B_inputs)
            recov_B = G_AB(fake_A_inputs)

            # Adversarial loss
            G_adv = -torch.mean(fake_A_outputs) - torch.mean(fake_B_outputs)
            # Cycle loss
            cycle_loss = torch.nn.L1Loss().to(device)
        
            G_cycle = cycle_loss(recov_A, img_A) + cycle_loss(recov_B, img_B)
            # Total loss
            lambda_adv = 1
            lambda_cycle = 10
            g_loss = lambda_adv * G_adv + lambda_cycle * G_cycle

            # Backward propagation
            G_AB_optimizer.zero_grad()  # Zero the parameter gradients
            G_BA_optimizer.zero_grad()
            g_loss.backward()
            G_AB_optimizer.step()  # Update D's parameters
            G_BA_optimizer.step()  # Update D's parameters

        if epoch % 1 == 0:
            if times % 51 == 0 or times == len(train_loader):
                print('[{}/{}, {}/{}] D_loss: {:.4f}  G_loss: {:.4f}'.format(epoch, args.epochs, times,
                                                                             len(train_loader), d_loss.item(),
                                                                             g_loss.item()))

    ############################
    # for visualization
    ###########################
    # # store generated images. Check how the generator is doing by saving G's output on fixed_noise
    if epoch % args.image_show_interval == 0:

        with torch.no_grad():
            fake = G_AB(img_A)
            fake = torch.mul(img_A, fake) # new loss
            X_fake = torch.cat([real_inputs, fake], dim=1)
        img_list = To_Image(img_list, X_fake)  # store all figures in img_list

        # Show generated images. Note: image_show_interval should be no smaller than image_store_interval
        Show_GeneratedImages(img_list)

    # Save Losses for plotting later
    G_losses.append(g_loss.item())
    D_losses.append(d_loss.item())

    if epoch % args.mode_save_interval == 0:
        # # Show final image comparision
        # Show_ImagesComparision(device, train_loader, img_list)
        # # save network during the training
        torch.save(G_AB.state_dict(), args.model_root + args.method + args.case_train + '/G_AB.pth')
        torch.save(G_BA.state_dict(), args.model_root + args.method + args.case_train + '/G_BA.pth')
        torch.save(D_A.state_dict(), args.model_root + args.method + args.case_train + '/D_A.pth')
        torch.save(D_B.state_dict(), args.model_root + args.method + args.case_train + '/D_B.pth')
        print('Model saved')
    if epoch == args.epochs:
        # # after training, write the losses to a .txt file
        LossVsEpoch(G_losses, D_losses)  # Loss versus training iteration
        LossWrite(G_losses, D_losses)

print('Training Finished.')
print('Cost Time: {}s'.format(time.time() - start_time))

Save_TrainingVideo(args.image_store_interval, img_list)