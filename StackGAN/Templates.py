import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from PIL import Image

"""
Ng: (Cond. Dim)Paper
Nz: (Noise Dim.)Paper 
Np: Text Embedding Dim (Reed et al)
Wo : Paper
Ho : Paper
"""

# Generated image might not be of size Wo Ho fix it ----
class G1(nn.Module):
    def __init__(self,Ng,Nz,Np):
        super(G1,self).__init__()
        self.ng = Ng # conditional dim: 128
        self.nz = Nz # Noise: 100
        self.np = Np # text embed: 1024

        self.ca_layer = nn.Sequential(
            nn.Linear(in_features= self.np, out_features= 2*self.ng, bias=True),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.ng + self.nz , self.ng * 4 * 4, bias = False),
            nn.BatchNorm1d(self.ng*4*4),
            nn.ReLU(True)
        )

        self.ups = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(self.ng, self.ng//2, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.ng//2),
            nn.ReLU(True),

            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(self.ng//2, self.ng//4, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.ng//4),
            nn.ReLU(True),

            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(self.ng//4, self.ng//8, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.ng//8),
            nn.ReLU(True),

            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(self.ng//8, self.ng//16, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.ng//16),
            nn.ReLU(True),

            nn.Conv2d(self.ng//16, 3, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.Tanh()
        )
        

    def ca_net(self,text_embed):
        x = self.ca_layer(text_embed)
        mu = x[:,:self.ng]
        logvar = x[:,self.ng:]
        std = logvar.mul(0.5).exp_()
        if (text_embed.is_cuda) :
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return mu, logvar, eps.mul(std).add_(mu)
    
    def forward(self,text_embed,noise):
        mu, logvar, c_code = self.ca_net(text_embed)
        z_c_code = torch.cat([noise, c_code],1)
        h_code = self.fc1(z_c_code) # vector 2 matrix

        h_code = h_code.view(-1,self.ng,4,4)
        gen_img = self.ups(h_code)
        return None, gen_img, mu, logvar


# ndf = Nd / 8


class D1(nn.Module):
    def __init__(self,Nd,Np):
        super(D1, self).__init__()
        self.nd = Nd
        self.np = Np # Here,Np refers to compressed embedding which is not 1024, but 128 (mu)
        
        self.encode_img = nn.Sequential(
            nn.Conv2d(3,self.nd//8, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(self.nd//8,self.nd//4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.nd//4),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(self.nd//4,self.nd//2, 4,2,1, bias = False),
            nn.BatchNorm2d(self.nd//2),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(self.nd//2,self.nd, 4,2,1, bias = False),
            nn.BatchNorm2d(self.nd),
            nn.LeakyReLU(0.2, inplace = True),
        )

        self.get_cond_logits = D_GET_LOGITS(self.nd, self.np)
        self.get_uncond_logits = None

    def forward(self,image):
        img_embed = self.encode_img(image)
        return img_embed
        

class D_GET_LOGITS(nn.Module):
    def __init__(self,Nd,Np,bcond = True):
        super(D_GET_LOGITS, self).__init__()
        self.nd = Nd
        self.np = Np 
        self.bcond = bcond
        if bcond :
            self.compressEmbed = nn.Sequential(
                nn.Linear(in_features= self.np, out_features= self.nd, bias=True),
                nn.ReLU()
            )
            self.outlogits = nn.Sequential(
                nn.Conv2d(self.nd + self.nd , self.nd, kernel_size = 3, stride = 1 , padding = 1, bias = False),
                nn.BatchNorm2d(self.nd),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.nd, 1, kernel_size=4, stride=4),
                # nn.Sigmoid()
            )
        else: 
            self.outlogits = nn.Sequential(
                nn.Conv2d(self.nd, 1 , kernel_size = 4, stride = 4),
                # nn.Sigmoid()
            )
    
    def forward(self,h_code , tex_embed = None):
        if self.bcond :
            c_code = self.compressEmbed(tex_embed)
            c_code = c_code.view(-1, self.nd, 1 , 1)
            c_code = c_code.repeat(1,1,4,4) # Md , Md
            h_c_code = torch.cat([h_code , c_code],1)
        else:
            h_c_code = h_code
        output = self.outlogits(h_c_code)
        return output.view(-1)

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            nn.Conv2d(channel_num, channel_num, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(channel_num),
        )
    def forward(self,x):
        residual = x
        out = self.block(x)
        out += residual 
        out = nn.functional.relu(out)
        return out


# ngf = np /8 --here 
# res_n: number of residual layers
class G2(nn.Module):
    def __init__(self,Ng, Np,Nz,res_n,Stage1_G = None):
        super(G2, self).__init__()
        self.ng = Ng
        self.np = Np
        self.nz = Nz
        self.stg1_G = Stage1_G
        if self.stg1_G == None:
            self.stg1_G = G1(Ng = Ng, Np = Np, Nz = Nz)

        for param in self.stg1_G.parameters():
            param.requires_grad = False
        
        self.ca_layer = nn.Sequential(
            nn.Linear(in_features= self.np, out_features= 2*self.ng, bias=True),
            nn.ReLU()
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.np//8, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.ReLU(True),

            nn.Conv2d(self.np//8, self.np//4, kernel_size = 4, stride = 2 , padding = 1, bias = False),
            nn.BatchNorm2d(self.np//4),
            nn.ReLU(True),

            nn.Conv2d(self.np//4 , self.np // 2 , 4 , 2 , 1 , bias = False),
            nn.BatchNorm2d(self.np//2),
            nn.ReLU(True)            
        )
        self.hr_joint = nn.Sequential(
            nn.Conv2d(self.ng + self.np//2, self.np//2, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.np//2),
            nn.ReLU(True)
        )
        # --------------------- Residuals 
        layers = []
        for i in range(res_n):
            layers.append(ResBlock(self.np//2))
        self.res_layers = nn.Sequential(*layers)
        # ---------------------

        self.ups = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(self.np//2, self.np//4, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.np//4),
            nn.ReLU(True),

            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(self.np//4, self.np//8, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.np//8),
            nn.ReLU(True),

            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(self.np//8, self.np//16 , kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.np//16),
            nn.ReLU(True),

            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(self.np//16, self.np//16, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.np//16),
            nn.ReLU(True),

            nn.Conv2d(self.np//16, 3, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.Tanh()
        )
        
    def ca_net(self,text_embed):
        x = self.ca_layer(text_embed)
        mu = x[:,:self.ng]
        logvar = x[:,self.ng:]
        std = logvar.mul(0.5).exp_()
        if (text_embed.is_cuda) :
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return mu, logvar, eps.mul(std).add_(mu)
    
    def forward(self,text_embed, noise):
        _, gen_img_1, _, _  = self.stg1_G(text_embed,noise)
        gen_img_1 = gen_img_1.detach()

        encoded_img = self.encoder(gen_img_1)

        c_code, mu, logvar = self.ca_net(text_embed)
        c_code = c_code.view(-1, self.ng,1,1)
        c_code = c_code.repeat(1,1,16,16)
        i_c_code = torch.cat([encoded_img, c_code],1)
        h_code = self.res_layers(self.hr_joint(i_c_code))

        gen_img_2 = self.ups(h_code)

        return gen_img_1, gen_img_2, mu, logvar

class D2(nn.Module):
    def __init__(self,Nd,Np):
        super(D2,self).__init__()
        self.nd = Nd
        self.np = Np
    
        self.encode_img = nn.Sequential(
            nn.Conv2d(3,self.nd//2, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2,inplace = True),
            
            nn.Conv2d(self.nd//2,self.nd, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.nd),
            nn.LeakyReLU(0.2,inplace = True),

            nn.Conv2d(self.nd,self.nd*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.nd*2),
            nn.LeakyReLU(0.2,inplace = True),

            nn.Conv2d(self.nd*2,self.nd*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.nd*4),
            nn.LeakyReLU(0.2,inplace = True),

            nn.Conv2d(self.nd*4,self.nd*8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.nd*8),
            nn.LeakyReLU(0.2,inplace = True),
            
            nn.Conv2d(self.nd*8,self.nd*16, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.nd*16),
            nn.LeakyReLU(0.2,inplace = True),
            
            nn.Conv2d(self.nd*16, self.nd*8, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.nd*8),
            nn.LeakyReLU(0.2,inplace = True),

            nn.Conv2d(self.nd*8, self.nd*4, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.nd*4),
            nn.LeakyReLU(0.2,inplace = True),

            nn.Conv2d(self.nd*4, self.nd*2, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.nd*2),
            nn.LeakyReLU(0.2,inplace = True),

            nn.Conv2d(self.nd*2, self.nd, kernel_size = 3, stride = 1 , padding = 1, bias = False),
            nn.BatchNorm2d(self.nd),
            nn.LeakyReLU(0.2,inplace = True)
        )
        
        self.get_cond_logits = D_GET_LOGITS(self.nd , self.np, bcond=True)
        self.get_uncond_logits = D_GET_LOGITS(self.nd, self.np, bcond= False)

    def forward(self,image):
        img_embed = self.encode_img(image)
        return img_embed

