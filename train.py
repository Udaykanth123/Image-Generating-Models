from random import shuffle
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Generator,Discriminator,weights_init
import argparse
import numpy as np
import torch.optim as opt
from torch.autograd import Variable
from dataset import load_data, normalize
from torchvision.utils import save_image


parser=argparse.ArgumentParser("DC_GAN")
parser.add_argument("--img_path",default="./bitmojis")
parser.add_argument("--batchsize",default=128)
parser.add_argument("--epochs",default=10)


def main():
    args=parser.parse_args()
    device="cuda:3"
    img_size=64
    z_size=100
    in_channels=3
    k=2
    model_G=Generator(z_size,img_size,args.batchsize).to(device)
    model_D=Discriminator(in_channels,args.batchsize).to(device)
    model_D.apply(weights_init)
    model_G.apply(weights_init)
    loss=nn.BCELoss().to(device)
    g_opt=opt.Adam(model_G.parameters(),lr=0.0002, betas=(0.5,0.999))
    d_opt=opt.Adam(model_D.parameters(),lr=0.0002, betas=(0.5,0.999))
    data=load_data(args.img_path)
    real_data=DataLoader(dataset=data,batch_size=args.batchsize,shuffle=True)
    gen_loss=[]
    dis_loss=[]
    fixed_noise = Variable(torch.FloatTensor(np.random.randn(args.batchsize,z_size,1,1))).to(device)
    for epoch in range(args.epochs):
        real_images=tqdm(real_data)
        model_G.train()
        model_D.train()
        loss_g=[]
        loss_d=[]
        batch_id=0
        for imgs in real_images:
            batch_id+=1
            z = Variable(torch.FloatTensor(np.random.randn(args.batchsize,z_size,1,1))).to(device)
            real_imgs=Variable(imgs).to(device)
            real=Variable(torch.FloatTensor(imgs.shape[0]).uniform_(0.8,1.2), requires_grad=False).to(device)
            fake=Variable(torch.FloatTensor(imgs.shape[0]).uniform_(0.0,0.2), requires_grad=False).to(device)
            for i in range(k):
                model_G.zero_grad()
                model_D.zero_grad()
                fake_imgs=model_G(z).to(device)
                fake_out=model_D(fake_imgs).to(device)
                real_out=model_D(real_imgs).to(device)
                d_loss=loss(fake_out,fake)+loss(real_out,real)
                # d_loss=1-real_out.mean()+fake_out.mean()
                d_loss.backward(retain_graph=True)
                d_opt.step()
             
            # training generator

            model_G.zero_grad()
            fake_imgs=model_G(z).to(device)
            fake_out=model_D(fake_imgs).to(device)
            g_loss=loss(fake_out,real)
            # g_loss=1-fake_out.mean()
            g_loss.backward()
            g_opt.step()
            loss_g.append(g_loss.item())
            loss_d.append(d_loss.item())
            if(batch_id%100==0):
                save_image((model_G(fixed_noise).data[:25]), "outputs/%d_%d.png" % (batch_id,epoch), nrow=5,normalize=True)
            real_images.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch,args.epochs,d_loss.item(),g_loss.item(),real_out.mean(),fake_out.mean()))


        gen_loss.append(loss_g)
        dis_loss.append(loss_d)
        if(epoch % 1 ==0):
            torch.save(model_G.state_dict(),"saving/gen/gen_{}.pth".format(epoch))
            # torch.save(model_D.state_dict(),"saving/disc/disc_{}.pth".format(epoch))



    np.savetxt("G_loss.csv",
           gen_loss,
           delimiter=", ",
           fmt='% s')
    np.savetxt("D_loss.csv",
           dis_loss,
           delimiter=", ",
           fmt='% s')

        
if __name__=="__main__":
    main()