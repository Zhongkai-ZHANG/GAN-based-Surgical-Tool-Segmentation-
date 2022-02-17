import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import save_image
from parameters import TrainParameters

# Train parameters
args = TrainParameters().parse()


# Plot one batch training images
def Show_ImageDataSet(device, data_loader):
    real_batch = next(iter(data_loader))  # Get a single batch from DataLoader without iterating
    plt.figure(figsize=(80, 80))
    plt.axis("off")
    plt.title("Training Images", fontsize=20)
    # make_grid的作用是将若干幅图像拼成一幅图像。其中padding的作用就是子图像与子图像之间的pad有多宽,
    # nrow来修改每一行的图片的数量; pad_value表示图片之间填充的颜色
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device), nrow=6, padding=2,
                                             normalize=True, pad_value=0.5).cpu(), (1, 2, 0)))
    plt.show()


def To_Image(img_list, InputTensor):  # 将数据转化为图片
    # 因为 PyTorch 的自动求导系统不会追踪 tensor.data 的变化，所以使用它的话可能会导致求导结果出错。
    # 官方建议使用 tensor.detach() 来替代它，二者作用相似，但是 detach 会被自动求导系统追踪，使用起来很安全
    output = InputTensor.detach().cpu().view(-1, args.image_channels, args.image_H, args.image_W)
    img_list.append(vutils.make_grid(output, nrow=12, padding=2,
                                     normalize=True, pad_value=0.5))
    return img_list


def Show_GeneratedImages(img_list):
    plt.figure(figsize=(50, 50))
    plt.axis("off")
    plt.title("Generated Images", fontsize=20)
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


# take a look at some real images and fake images side by side. At the same time, save images
def Show_ImagesComparision(device, data_loader, img_list):
    # Grab a batch of real images from the DataLoader
    real_batch = next(iter(data_loader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images", fontsize=20)
    TrainingImage = np.transpose(vutils.make_grid(real_batch[0].to(device)[:16], nrow=4, padding=5, normalize=True)
                                 .cpu(), (1, 2, 0))
    plt.imshow(TrainingImage)

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Generated Images", fontsize=20)
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


# 现在，我们可以用动画来可视化G的训练进度.
def Save_TrainingVideo(image_store_interval, img_list):
    # Note: bug: save_image(img_list[-1], "images/..0.png") can only use once
    for n in range(0, len(img_list)):  # save generated images
        name = "/content/drive/My Drive/Autoencoder/images/GeneratedImage_" + str(image_store_interval * n) + ".png"
        save_image(img_list[n], name)


def LossVsEpoch(G_losses):  # 下面是迭代过程中 D 与 G 的损失对比图。
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training", fontsize=20)
    plt.plot(G_losses, label="G")
    plt.xlabel("iterations", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.legend()
    plt.show()


def LossWrite(G_losses):  # write all losses to .txt file
    np.savetxt(args.losses_root + '/G_losses.txt', G_losses)
