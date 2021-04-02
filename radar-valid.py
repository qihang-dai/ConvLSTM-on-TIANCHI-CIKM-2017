#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.radar import Radar
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import argparse
import torchvision
#APEX FT16加速
from torch.cuda.amp import autocast as autocast, GradScaler
from utils import create_video






def test(check,model,TIMESTAMP):
    '''
    main function to run the training
    '''

    parser = argparse.ArgumentParser()
    ##可选参数：如python demo.py --family=张 --name=三
    parser.add_argument('-clstm',
                        '--convlstm',
                        help='use convlstm as base cell',
                        action='store_true')
    parser.add_argument('-cgru',
                        '--convgru',
                        help='use convgru as base cell',
                        action='store_true')
    parser.add_argument('--batch_size',
                        default=4,
                        type=int,
                        help='mini-batch size')
    parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
    parser.add_argument('-frames_input',
                        default=7,
                        type=int,
                        help='sum of input frames')
    parser.add_argument('-frames_output',
                        default=7,
                        type=int,
                        help='sum of predict frames')
    parser.add_argument('-epochs', 
                        default=20, #设置epoch
                        type=int, 
                        help='sum of epochs')
    args = parser.parse_args()

    random_seed = 1996
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    save_dir = './save_model/' + TIMESTAMP

    validFolder = Radar(is_train=False,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
    validLoader = torch.utils.data.DataLoader(validFolder,
                                            batch_size=args.batch_size,
                                            shuffle=False,num_workers=4,pin_memory=True)
    print(len(validLoader))

    if model=='lstm':
        encoder_params = convlstm_encoder_params
        decoder_params = convlstm_decoder_params
        print("convlstm",flush=True)
    elif model=='gru':
        encoder_params = convgru_encoder_params
        decoder_params = convgru_decoder_params
        print('convgru1',flush=True)
    else:
        # encoder_params = convgru_encoder_params
        # decoder_params = convgru_decoder_params
        print('default-convlstm',flush=True)
        encoder_params = convlstm_encoder_params
        decoder_params = convlstm_decoder_params


    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device) 
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    #check point断点续传
    
    if os.path.exists(os.path.join(save_dir, check)):
        # load existing model
        print('==> loading specific epoch model')
        model_info = torch.load(os.path.join(save_dir, check))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    # else:
    #     if not os.path.isdir(save_dir):
    #         os.makedirs(save_dir)
    #     cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)  #half the learning rate every 4 epoch without improving. No print message.






    with torch.no_grad():
        net.eval()
        t = tqdm(validLoader, leave=False, total=len(validLoader))
        print("Test=====================",end="")
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            
            inputs = inputVar.to(device)
            label = targetVar.to(device)
            with autocast():
                pred = net(inputs)
                loss = lossfunction(pred, label)
            if i % 50 == 0:
                image=create_video(inputs,pred,label)
                tb.add_image('Try'+str(i), image, 0)
                tb.close()
            if i == 151:
                break
        torch.cuda.empty_cache()


if __name__ == "__main__":
    check='checkpoint_20_0.002446.pth'+".tar"
    model='gru'
    TIMESTAMP = "2021-03-28"
    test(check,model,TIMESTAMP)

#tensorboard --logdir='D:\download\ConvLSTMpytorch\Radar\runs'
