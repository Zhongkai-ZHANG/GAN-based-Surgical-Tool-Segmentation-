import torch
import torch.nn as nn
import torch.autograd as autograd


# user-defined loss function: standard L1 loss
class MyL1Loss(nn.Module):
    def __init__(self):
        super(MyL1Loss, self).__init__()

    def forward(self, Generate, Original):
        # to train on CPU, make sure to call contiguous() on x and y before reshaping
        x = Generate.view(-1, 1)  # for linux, x = Generate.view(-1, 1)
        y = Original.view(-1, 1)  # for linux, y = Original.view(-1, 1)
        # print(x[2][0])
        # print(x.size(0))
        # print(x.shape)
        # 对于在求梯度阶段需要用到的张量不能使用 inplace operation. So put Error in front of inplace operation
        Error = torch.abs(x - y) 
        
        # yNumpy = y.cpu().detach().numpy()
        # ErrorNumpy = Error.cpu().detach().numpy()
        # for i in range(yNumpy.size//3):  # 在Python3里：/的结果是真正意义上的除法，结果是float型。用双//就可以了
        #  # if 1.01 > xNumpy[3*i][0] > 0.99 and 1.01 > xNumpy[3*i + 1][0] > 0.99 and 1.01 > xNumpy[3*i + 2][0] > 0.99:
        #     if yNumpy[3*i][0] == 1 and yNumpy[3*i + 1][0] == 1 and yNumpy[3*i + 2][0] == 1:
        #         ErrorNumpy[3*i][0] = 0
        #         ErrorNumpy[3 * i + 1][0] = 0
        #         ErrorNumpy[3 * i + 2][0] = 0
        # #  转换后的tensor与numpy指向同一地址，所以，对一方的值改变另一方也随之改变
        # Error.cuda()
        
        loss = torch.mean(Error)
        # loss = torch.mean(torch.abs(x - y).pow(2))
        return loss


# Optimizers
def Get_optimizers(args, G1, G2, D1, D2):
    G1_optimizer = torch.optim.Adam(G1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    G2_optimizer = torch.optim.Adam(G2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    D1_optimizer = torch.optim.Adam(D1.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    D2_optimizer = torch.optim.Adam(D2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    return G1_optimizer, G2_optimizer, D1_optimizer, D2_optimizer


def compute_gradient_penalty(D, real_data, generated_data):
    """Calculates the gradient penalty loss for WGAN GP"""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Random weight term for interpolation between real and fake data
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1)  #  matrix with dimension batch_size*1*1*1
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)

    # Get random interpolation between real and fake samples
    interpolation = (alpha * real_data.data + (1 - alpha) * generated_data.data).requires_grad_(True)
    interpolation = interpolation.to(device)

    interpolation_logits = D(interpolation)
    grad_outputs = torch.ones(interpolation_logits.size())
    grad_outputs = grad_outputs.to(device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=interpolation_logits,
                              inputs=interpolation,
                              grad_outputs=grad_outputs,
                              create_graph=True,
                              retain_graph=True)[0]
    # Gradients have shape (batch_size, num_channels, img_width, img_height)
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    # Return gradient penalty
    return gradient_penalty
