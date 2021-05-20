import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import h5py
import torch 
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import io

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.img_size = 64
        self.nc = 3
        self.ndim = 100
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.latent_dim = self.ndim + self.projected_embed_dim
        self.ngf = 64

        self.netProject = nn.Sequential(
            nn.Linear(in_features= self.embed_dim, out_features= self.projected_embed_dim),
            nn.BatchNorm1d(num_features = self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2,inplace = True)
        )

        self.netG = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self,embed ,noise):
        embed_projected = self.netProject(embed).unsqueeze(2).unsqueeze(3)
        latent = torch.cat([embed_projected,noise], 1)
        return self.netG(latent)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        """ model_type: 0-> DCGAN , 1-> WGAN
        """

        self.img_size = 64
        self.nc = 3
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.ndf = 64
        
        self.projector = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.netD1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.netD2 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
       
          

    def forward(self, input,embed):
        x_int = self.netD1(input)
        x = self.projector(embed)
        rep_embed = x.repeat(4, 4, 1, 1).permute(2,  3, 0, 1)
        x = torch.cat([x_int, rep_embed], 1)
        x = self.netD2(x)
        return x.view(-1,1).squeeze(1),x_int  


class CustomDataset(Dataset):
    def __init__(self,file,splitType,transform=None):
        # splitType: train, test, val
        self.file = file
        self.transform = transform
        f = h5py.File(self.file,'r')
        self.dataset = f[splitType] 
        self.dkeys = list(self.dataset)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,index):
        inst_name = self.dkeys[index]
        inst = self.dataset[inst_name]

        rImg = (np.array(Image.open(io.BytesIO(np.array(inst['img']).tobytes())).convert("RGB").resize((64,64),resample = 3),dtype = np.float32) - 127.5)/127.5
        rImg = rImg.transpose(2,0,1)
        
        rEmbed = np.array(inst['embeddings'],dtype = float)
        randIndex = random.randint(0,len(self.dkeys)-1)
        while inst['class'][()] == self.dataset[self.dkeys[randIndex]]['class'][()]:
            randIndex = random.randint(0,len(self.dkeys)-1)    
        wImg = (np.array(Image.open(io.BytesIO(np.array(self.dataset[self.dkeys[randIndex]]['img']).tobytes())).convert("RGB").resize((64,64),resample = 3),dtype=np.float32) - 127.5)/127.5
        wImg = wImg.transpose(2,0,1)
        randIndex = random.randint(0,len(self.dkeys)-1)
        wEmbed = np.array(self.dataset[self.dkeys[randIndex]]['embeddings'],dtype = np.float32)


        txt = inst['txt'][()]
        special = u"\ufffd\ufffd"
        # txt = txt.replace(special,' ')
        txt = str(np.array(txt).astype(str))

        dataPoint = {
                    'rImg': torch.FloatTensor(rImg),
                    'wImg': torch.FloatTensor(wImg),
                    'rEmbed': torch.FloatTensor(rEmbed),
                    'wEmbed': torch.FloatTensor(wEmbed),
                    'txt': txt 
                    }
        return dataPoint


