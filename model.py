import torch
from torch import nn


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)



class Generator(nn.Module):
    def __init__(self,z_size,output_size,batch_size):
        super(Generator,self).__init__()
        k=output_size//16
        self.block1=nn.ConvTranspose2d(z_size, 1024,kernel_size=4, stride=1)
        self.batch_size=batch_size
        self.block2=nn.Sequential(
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.ConvTranspose2d(1024,512,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64,3,kernel_size=3,padding=1),
        nn.Tanh()
        )
    
    def forward(self,x):
        x=self.block1(x)
        # x=x.view(x.shape[0],1024,4,4)
        return self.block2(x)






class Discriminator(nn.Module):
    def __init__(self,in_channels,batch_size):
        super(Discriminator,self).__init__()
        self.batch_size=batch_size
        self.block=nn.Sequential(
            # nn.BatchNorm2d(1)
            nn.Conv2d(in_channels,256,kernel_size=4,padding=2,stride=2), # 64*64
            # nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(256,128,kernel_size=4,padding=2,stride=2), #32*32
            # nn.MaxPool2d(kernel_size=8,stride=8),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128,64,kernel_size=4,padding=2,stride=2), #16*16
            # nn.MaxPool2d(kernel_size=8,stride=8),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64,32,kernel_size=4,padding=2,stride=2), #8*8
            # nn.MaxPool2d(kernel_size=4,stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32,16,kernel_size=4,padding=2,stride=2), #4*4
            # nn.MaxPool2d(kernel_size=4,stride=4),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16,8,kernel_size=4,padding=2,stride=4), #1*1
            # nn.MaxPool2d(kernel_size=4,stride=4),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(128,1024,kernel_size=1),
            # nn.LeakyReLU(0.2),

            nn.Conv2d(8,1,kernel_size=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        batch_size = x.size(0)
        return torch.sigmoid(self.block(x).view(batch_size))