import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG,NetD
import torchvision.utils as vutils

from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from DAMSM import RNN_ENCODER

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)
import matplotlib.pyplot as plt

UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./FY-GAN/code/cfg/fashion.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

###For generating images for testing dataset
def sampling(text_encoder, netG, dataloader,device,f,filename,model_dir):
  #load the model  
  netG.load_state_dict(torch.load('%s'%(f))['model_state_dict'])
  netG.eval()
  batch_size = cfg.TRAIN.BATCH_SIZE
  s_tmp = model_dir
  save_dir = '%s/%s' % (s_tmp, filename[0:-4])
  #print(save_dir)
  mkdir_p(save_dir)
  cnt = 0
  for i in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
      for step, data in enumerate(dataloader, 0):
          imags, captions, cap_lens, class_ids, keys = prepare_data(data)
          cnt += batch_size
          if step % 100 == 0:
              print('step: ', step)
          hidden = text_encoder.init_hidden(batch_size)
          # words_embs: batch_size x nef x seq_len
          # sent_emb: batch_size x nef
          words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
          words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
          #######################################################
          # (2) Generate fake images
          ######################################################
          with torch.no_grad():
              noise = torch.randn(batch_size, 100)
              noise=noise.to(device)
              fake_imgs = netG(noise,sent_emb)
          for j in range(batch_size):
              s_tmp = '%s/%s' % (save_dir, keys[j])
              folder = s_tmp[:s_tmp.rfind('/')]
              if not os.path.isdir(folder):
                  print('Make a new folder: ', folder)
                  mkdir_p(folder)
              im = fake_imgs[j].data.cpu().numpy()
              # [-1, 1] --> [0, 255]
              im = (im + 1.0) * 127.5
              im = im.astype(np.uint8)
              im = np.transpose(im, (1, 2, 0))
              im = Image.fromarray(im)
              fullpath = '%s_%3d.png' % (s_tmp,i)
              im.save(fullpath)


###For generating images for training dataset
def train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD,state_epoch,batch_size,device):
    #loss_file = open("./FY-GAN/losses8.txt", "w+")
    for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH+1):    
        for step, data in enumerate(dataloader, 0):
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            all_cap=[]
            for key in keys:
              filepath = '%s/text/%s.txt' % ('./FY_GAN/data/fashion', key)
              with open(filepath, "r",encoding="utf8", errors='ignore') as f:
                c=f.readlines()             
                cnt = 0
                for cap in c:
                    all_cap.append(cap.strip())
                    cnt += 1
                    if cnt == cfg.TEXT.CAPTIONS_PER_IMAGE:
                        break
                if cnt < cfg.TEXT.CAPTIONS_PER_IMAGE:
                    print('ERROR: the captions for %s less than %d' % (key, cnt))
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            imgs=imags[0].to(device)
            real_features = netD(imgs)
            output = netD.COND_DNET(real_features,sent_emb)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()

            output = netD.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
            errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()

            # synthesize fake images
            noise = torch.randn(batch_size, 100)
            noise=noise.to(device)
            fake = netG(noise,sent_emb)  
            
            # G does not need update with D
            fake_features = netD(fake.detach()) 

            errD_fake = netD.COND_DNET(fake_features,sent_emb)
            errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()          
            #errD = errD_real - errD_fake #WGAN
            errD = errD_real + (errD_fake + errD_mismatch)/2.0
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            errD.backward()
            optimizerD.step()

            #MA-GP loss function 
            interpolated = (imgs.data).requires_grad_()
            sent_inter = (sent_emb.data).requires_grad_()
            features = netD(interpolated)
            out = netD.COND_DNET(features,sent_inter)
            grads = torch.autograd.grad(outputs=out,
                                    inputs=(interpolated,sent_inter),
                                    grad_outputs=torch.ones(out.size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0,grad1),dim=1)                        
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            d_loss.backward()
            optimizerD.step()
            
            # update G
            features = netD(fake)
            output = netD.COND_DNET(features,sent_emb)
            errG = - output.mean()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            errG.backward()
            optimizerG.step()
            loss_file.write('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f\n'
                % (epoch, cfg.TRAIN.MAX_EPOCH, step, len(dataloader), errD.item(), errG.item()))
            #printing loss
            print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
                % (epoch, cfg.TRAIN.MAX_EPOCH, step, len(dataloader), errD.item(), errG.item()))
        ifked not os.path.exists('imgs'):
          os.mairs('/content/drive/Shareddrives/ML: DF GAN for Fashion/FY-GAN/imgs')
        vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d.png' % ('./FY-GAN/imgs', epoch),
                        normalize=True)
         
        #storing captions of particular batch
        file_cap='./FY-GAN/caps/'+str(epoch)+'txt'
        with open(file_cap, 'w') as filehandle:
          for listitem in all_cap:
              filehandle.write('%s\n' % listitem)
        #saving models
        if not os.path.exists('models'):
          os.makedirs('models')
        if not os.path.exists('models/fashion'):
          os.makedirs('models/fashion')
        if epoch%5==0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': netG.state_dict(),
                'optimizer_state_dict': optimizerG.state_dict(),
                }, './FY-GAN/models/%s/netG_%03d.pth' % (cfg.CONFIG_NAME, epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                }, './FY-GAN/models/%s/netD_%03d.pth' % (cfg.CONFIG_NAME, epoch))     

    return count




if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        #args.manualSeed = random.randint(1, 10000)
    print("seed now is : ",args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    
     # # validation data #
    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:     
        
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
        
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)
    dataset.n_words = 5450
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    checkpoint = torch.load(cfg.TEXT.DAMSM_NAME) 
    text_encoder.load_state_dict(checkpoint['model_state_dict'])
    text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()    

    state_epoch=0

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))  
    # checkpoint = torch.load("./FY-GAN/models/bird/netG_030.pth")
    # netG.load_state_dict(checkpoint['model_state_dict'])
    # optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
    # state_epoch = checkpoint['epoch']
    # checkpoint2 = torch.load("./FY-GAN/models/bird/netD_030.pth")
    # netD.load_state_dict(checkpoint2['model_state_dict'])
    # optimizerD.load_state_dict(checkpoint2['optimizer_state_dict'])
    # state_epoch = checkpoint2['epoch']
    if cfg.B_VALIDATION:
            #storing images of different models
            model_dir = "FY-GAN/ensemble_images"
            # Directory of saved mpdels
            directory = 'FY-GAN/ensemble_models'
            for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
                count = sampling(text_encoder, netG, dataloader,device,f,filename,model_dir)  # generate images for the whole valid dataset
                print('state_epoch:  %d'%(state_epoch))
    else:
        count = train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD, state_epoch,batch_size,device)



        
