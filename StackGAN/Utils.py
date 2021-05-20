#-----------------------
import h5py
import torch 
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable
import argparse
import os
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import sys, os

from Templates import *

#----------------------
class CustomDataset(Dataset):
    def __init__(self,file,splitType,transform=None):
        # splitType: train, test, val
        self.file = file
        self.transform = transform
        f = h5py.File(self.file,'r')
        self.splitType = splitType 
        self.dataset = f[splitType] 
        self.dkeys = list(self.dataset)
        self.len = len(self.dataset) ; f.close()
    def __len__(self):
        return self.len
    def __getitem__(self,index):
        inst_name = self.dkeys[index]
        f = h5py.File(self.file,'r')
        self.dataset = f[self.splitType]
        inst = self.dataset[inst_name]

        rImg = (np.array(Image.open(io.BytesIO(np.array(inst['img']).tobytes())).convert("RGB").resize((64,64),resample = 3),dtype = np.float32) - 127.5)/127.5
        rImg = rImg.transpose(2,0,1)
        rImgB = (np.array(Image.open(io.BytesIO(np.array(inst['img']).tobytes())).convert("RGB").resize((256,256),resample = 3),dtype = np.float32) - 127.5)/127.5
        rImgB = rImgB.transpose(2,0,1)
        
        rEmbed = np.array(inst['embeddings'],dtype = float)
        randIndex = random.randint(0,len(self.dkeys)-1)
        while inst['class'][()] == self.dataset[self.dkeys[randIndex]]['class'][()]:
            randIndex = random.randint(0,len(self.dkeys)-1)    
        wImg = (np.array(Image.open(io.BytesIO(np.array(self.dataset[self.dkeys[randIndex]]['img']).tobytes())).convert("RGB").resize((64,64),resample = 3),dtype=np.float32) - 127.5)/127.5
        wImgB = (np.array(Image.open(io.BytesIO(np.array(self.dataset[self.dkeys[randIndex]]['img']).tobytes())).convert("RGB").resize((256,256),resample = 3),dtype=np.float32) - 127.5)/127.5
        wImg = wImg.transpose(2,0,1)
        wImgB = wImgB.transpose(2,0,1)
        randIndex = random.randint(0,len(self.dkeys)-1)
        wEmbed = np.array(self.dataset[self.dkeys[randIndex]]['embeddings'],dtype = np.float32)
        

        txt = inst['txt'][()]; f.close()
        # special = '\ufffd\ufffd"
        # txt = txt.replace(special,' ')
        txt = str(np.array(txt).astype(str))


        dataPoint = {
                    'rImg': torch.FloatTensor(rImg),
                    'rImgB': torch.FloatTensor(rImgB),
                    'wImg': torch.FloatTensor(wImg),
                    'wImgB': torch.FloatTensor(wImgB),
                    'rEmbed': torch.FloatTensor(rEmbed),
                    'wEmbed': torch.FloatTensor(wEmbed),
                    'txt': txt 
                    }
        return dataPoint

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def KL_Loss(mu, logvar):
    KLD_element =mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def netD_loss(netD, real_imgs , fake_imgs , real_labels,
                fake_labels, tex_embed):
    batch_size = real_imgs.size(0)
    
    criterion = nn.BCEWithLogitsLoss()
    embed = tex_embed
    fake = fake_imgs.detach()
    real_features = netD(real_imgs)
    fake_features = netD(fake)

    real_logits = netD.get_cond_logits(real_features,embed)
    errD_real = criterion(real_logits,real_labels)

    wrong_logits = netD.get_cond_logits(real_features[:(batch_size - 1)] , embed[1:]) 
    errD_wrong = criterion(wrong_logits,fake_labels[1:])


    fake_logits = netD.get_cond_logits(fake_features,embed)
    errD_fake = criterion(fake_logits,fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = netD.get_uncond_logits(real_features)
        fake_logits = netD.get_uncond_logits(fake_features)

        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)

        errD = ((errD_real + uncond_errD_real) /2. + 
                ((errD_fake + errD_wrong + uncond_errD_fake ))/3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.

    else:
        errD = (errD_real + errD_fake + errD_wrong)*0.5
    return errD, errD_real.data, errD_wrong.data, errD_fake.data

def netG_loss(netD, fake_imgs, real_labels, tex_embed):
    criterion = nn.BCEWithLogitsLoss()
    embed = tex_embed
    fake_features = netD(fake_imgs)
    fake_logits = netD.get_cond_logits(fake_features,embed)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.get_uncond_logits is not None:
        fake_logits = netD.get_uncond_logits(fake_features)
        uncond_errD_fake = criterion(fake_logits,real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake


def log_result(epoch, i, netDLoss, netGLoss):
    #, real_score , fake_score)
    print("Epoch %d, Iter: %d, netDLoss: %f, netGLoss: %f"%(
                    epoch, i, netDLoss.cpu().mean(), netGLoss.cpu().mean()))# real_score.cpu().mean(), fake_score.cpu().mean()                ))
                #, D(X): %f, D(G(X)): %f"

def load_models(path,params):
    netG, netD = None, None
    if (params.stage == 1):
        netG = G1(Ng = params.Ng,Np = params.Np,Nz = params.Nz )
        netD = D1(Nd = params.Nd, Np = params.Np)
        if (params.trained):
            model_dict = torch.load(path + '/G1.pth')
            netG.load_state_dict(model_dict['state_dict'])
            
            model_dict = torch.load(path + '/D1.pth')
            netD.load_state_dict(model_dict['state_dict'])
        else : 
            netD.apply(weights_init)
            netG.apply(weights_init)
    elif (params.stage == 2):
        netG1 = G1(Ng = params.Ng,Np = params.Np,Nz = params.Nz )
        model_dict = torch.load(path + 'G1.pth')
        netG1.load_state_dict(model_dict['state_dict'])

        netG = G2(Ng = params.Ng,Np = params.Np,Nz = params.Nz,res_n= params.res_n , Stage1_G = netG1 )
        netD = D2(Np = params.Np, Nd = params.Nd)

        if(params.trained):
            model_dict = torch.load(path + 'G2.pth')
            netG.load_state_dict(model_dict['state_dict'])
            
            model_dict = torch.load(path + 'D2.pth')
            netD.load_state_dict(model_dict['state_dict'])
        else : 
            netG.apply(weights_init)
            netD.apply(weights_init)
            netG.stg1_G.load_state_dict(netG1.state_dict())
            for p in netG.stg1_G.parameters():
                p.requires_grad = False
    return netG.to(params.device), netD.to(params.device)

def save_model(path, model_dict, fName):
    """ fName: G1 D1 G2 D2
    """ 
    torch.save(model_dict, path+ fName+ '.pth')
    print(fName +" saved @ "+ path+ fName+ '.pth' )

def show_results(data, params, modelG):
    rImg = 'rImgB' if params.stage == 2 else 'rImg' if params.stage == 1 else 0
    right_images = Variable(data[rImg].float()).to(params.device)
    right_embeds  = Variable(data['rEmbed'].float()).to(params.device)
    noise = torch.randn((right_images.size(0),params.Nz),device = params.device)
    _,gen_imgs , _, _ = modelG(right_embeds, noise)

    gen_imgs = gen_imgs.detach().cpu().numpy()
    right_images = right_images.detach().cpu().numpy()

    nImages = 6
    for i in range(nImages):
        GT = right_images[i].transpose(1,2,0)
        Gen = gen_imgs[i].transpose(1,2,0)
        plt.subplot(2,nexam,i+1)
        plt.imshow(GT)
        plt.subplot(2,nexam,i+1+nexam)
        plt.imshow(Gen)
    plt.show()