import torch
import torch.nn as nn
from parameters import TrainParameters

# Train parameters
args = TrainParameters().parse()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


################################################
#    U-NET: G U-Net（256*256）     ##
################################################
# U-Net encoder (ConvBlock)
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# U-Net decoder with skip-connect (DeconvBlock)
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(out_size),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        # def forward(self, x):
        #     x = self.model(x)
        #     # x = torch.cat((x, skip_input), 1)

        return x


class Generator(nn.Module):
    def __init__(self, in_channels=args.image_channels, out_channels=args.image_channels, ngf=args.ngf):
        super(Generator, self).__init__()

        self.down1 = UNetDown(in_channels, ngf, normalize=False)  # no need of BatchNorm # 256*256->128*128
        self.down2 = UNetDown(ngf, ngf * 2)  # 128*128->64*64
        self.down3 = UNetDown(ngf * 2, ngf * 4)  # 64*64->32*32
        self.down4 = UNetDown(ngf * 4, ngf * 8)  # 32*32->16*16
        self.down5 = UNetDown(ngf * 8, ngf * 8, dropout=0.5)  # 16*16->8*8
        self.down6 = UNetDown(ngf * 8, ngf * 8, dropout=0.5)  # 8*8->4*4
        self.down7 = UNetDown(ngf * 8, ngf * 8, dropout=0.5)  # 4*4->2*2
        self.down8 = UNetDown(ngf * 8, ngf * 8, normalize=False, dropout=0.5)  # 2*2->1*1
        
        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=0.5)  # 1*1->2*2
        self.up2 = UNetUp(ngf * 16, ngf * 8, dropout=0.5)  # 2*2->4*4
        self.up3 = UNetUp(ngf * 16, ngf * 8, dropout=0.5)  # 4*4->8*8
        self.up4 = UNetUp(ngf * 16, ngf * 8, dropout=0.5)  # 8*8->16*16
        self.up5 = UNetUp(ngf * 16, ngf * 4)  # 16*16->32*32
        self.up6 = UNetUp(ngf * 8, ngf * 2)  # 32*32->64*64
        self.up7 = UNetUp(ngf * 4, ngf)  # 64*64->128*128
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )  # 128*128->256*256

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.up8(u7)


# PatchGAN ## 判别器的模型架构中不能使用Batch Normalization
class Discriminator(nn.Module):
    def __init__(self, in_ch=args.image_channels, out_ch=args.image_channels, ndf=args.ndf):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.convolution = nn.Sequential(
            #  no need of BatchNorm 256*256->128*128
            # *discriminator_block(in_ch + out_ch, ndf, normalization=False),
            *discriminator_block(in_ch, ndf, normalization=False),  # delete condition for discriminator
            *discriminator_block(ndf, ndf * 2),  # 128*128->64*64
            *discriminator_block(ndf * 2, ndf * 4),  # 64*64->32*32
            *discriminator_block(ndf * 4, ndf * 8),  # 32*32->16*16
            *discriminator_block(ndf * 8, ndf * 16),  # 16*16->8*8
            *discriminator_block(ndf * 16, ndf * 32),  # 8*8->4*4
            nn.Conv2d(ndf * 32, 1, kernel_size=4, stride=2, padding=1),  # 4*4->1*1.
            # nn.Sigmoid()   # 14*14->1
        )

    def forward(self, X):
        x = self.convolution(X)
        return x


class DiscriminatorPix(nn.Module):
    def __init__(self, in_ch=args.image_channels, out_ch=args.image_channels, ndf=args.ndf):
        super(DiscriminatorPix, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.convolution = nn.Sequential(
            #  no need of BatchNorm 256*256->128*128
            *discriminator_block(in_ch + out_ch, ndf, normalization=False),
            # *discriminator_block(in_ch, ndf, normalization=False),  # delete condition for discriminator
            *discriminator_block(ndf, ndf * 2),  # 128*128->64*64
            *discriminator_block(ndf * 2, ndf * 4),  # 64*64->32*32
            *discriminator_block(ndf * 4, ndf * 8),  # 32*32->16*16
            *discriminator_block(ndf * 8, ndf * 16),  # 16*16->8*8
            *discriminator_block(ndf * 16, ndf * 32),  # 8*8->4*4
            nn.Conv2d(ndf * 32, 1, kernel_size=4, stride=2, padding=1),  # 4*4->1*1.
            # nn.Sigmoid()   # 14*14->1
        )

    def forward(self, X):
        x = self.convolution(X)
        return x
        
####################################################
# Initialize generator and discriminator
####################################################
def Create_nets(device, mode):
    G1 = Generator().to(device)
    G2 = Generator().to(device)
    D2 = Discriminator().to(device)
    if mode == 1 or mode == 5:
        D1 = DiscriminatorPix().to(device)
    else:
        D1 = Discriminator().to(device)

    if args.premodel == 1:  # load the last trained parameter
        if mode == 0:  # I2IS-1D
            G1.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G.pth'))
            G1.eval()
            D1.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/D.pth'))
            D1.eval()
        if mode == 1:  # I2IS-1cD
            G1.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G.pth'))
            G1.eval()
            D1.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/D.pth'))
            D1.eval()
        if mode == 2:  # I2IS-2D
            G1.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G.pth'))
            G1.eval()
            D1.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/D_M.pth'))
            D1.eval()
            D2.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/D_L.pth'))
            D2.eval()
        if mode == 3:  # dualGAN
            G1.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G_AB.pth'))
            G1.eval()
            G2.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G_BA.pth'))
            G2.eval()
            D1.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/D_A.pth'))
            D1.eval()
            D2.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/D_B.pth'))
            D2.eval()
        if mode == 4:  # supervised learning
            G1.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G.pth'))
            G1.eval()
        if mode == 5:  # pix2pix
            G1.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G.pth'))
            G1.eval()
            D1.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/D.pth'))
            D1.eval()


    if args.premodel == 0:
        G1.apply(weights_init)
        G2.apply(weights_init)
        D1.apply(weights_init)
        D2.apply(weights_init)

    return G1, G2, D1, D2


def GetG_net(device, mode):  # for the test
    G = Generator().to(device)

    if mode == 0:  # I2IS-1D
        G.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G.pth'))
        G.eval()
    if mode == 1:  # I2IS-1cD
        G.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G.pth'))
        G.eval()
    if mode == 2:  # I2IS-2D
        G.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G.pth'))
        G.eval()
    if mode == 3:  # dualGAN
        G.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G_AB.pth'))
        G.eval()
    if mode == 4:  # supervised learning
        G.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G.pth'))
        G.eval()
    if mode == 5:  # pix2pix
        G.load_state_dict(torch.load(args.model_root + args.method + args.case_train + '/G.pth'))
        G.eval()
    
    return G
