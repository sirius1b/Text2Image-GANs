import h5py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from src.model_store import *

import warnings
warnings.filterwarnings('ignore')

def get_dataset():
    cwd = os.getcwd()
    file = cwd+'/dataset/flowers.hdf5'
    dSet = CustomDataset(file = file, splitType='train')
    loader = DataLoader(dSet, batch_size=6, shuffle=True, num_workers=0)
    return loader
 
def load_model(flag):
    if flag == 1:
        filename = './modelState/dcgan_cls.pth'
    elif flag == 2:
        filename = './modelState/gan_cls_int.pth'
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netG.apply(weights_init)

    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(weights_init)

    param_dict = torch.load(filename)
    netD.load_state_dict(param_dict['netD_state'])
    netG.load_state_dict(param_dict['netG_state'])
    return netG, netD, device


def get_results(netG, device, loader):
    val = next(enumerate(loader))[1]
    imgTrue = val['rImg'].numpy()
    captions = val['txt']
    nexam = 6
    with torch.no_grad():
        noise = torch.randn((val['rImg'].size(0),100),device = device)
        noise = noise.view(noise.size(0),100,1,1).to(device)
        fake_images = netG(Variable(val['rEmbed'].float()).to(device),noise).cpu().numpy()

    plt.rcParams["figure.figsize"] = (20,5)
    for i in range(nexam):
        GT = imgTrue[i].transpose(1,2,0)
        Gen = fake_images[i].transpose(1,2,0)
        plt.subplot(2,nexam,i+1)
        plt.imshow(GT)
        plt.subplot(2,nexam,i+1+nexam)
        plt.imshow(Gen)
    print("[+] Captions")
    for i in range(nexam):
        print(val['txt'][i])
    plt.show()

if __name__ == '__main__':
    print("\n[+] Getting the dataset")
    loader = get_dataset()
    print("[+] Dataset Loaded")
    print("\n[+] DCGAN-CLS")
    print("[+] Loading Model")
    netG, netD, device = load_model(1)
    print("[+] Model Loaded")
    print("[+] Generating Results")
    get_results(netG, device, loader)
    print("[+] DCGAN-CLS Finished")
    print("\n[+] GAN-CLS-INT")
    print("[+] Loading Model")
    netG, netD, device = load_model(2)
    print("[+] Model Loaded")
    print("[+] Generating Results")
    get_results(netG, device, loader)
    print("[+] GAN-CLS-INT Finished")
    print("\n[+] Exit")