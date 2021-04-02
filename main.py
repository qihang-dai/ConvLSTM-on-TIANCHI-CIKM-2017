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



TIMESTAMP = "2021-04-01-GRU"
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

# [Modify position] Change your dataset here

trainFolder = Radar(is_train=True,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
validFolder = Radar(is_train=False,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False,num_workers=4,pin_memory=True)
validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False,num_workers=4,pin_memory=True)


if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
    print("convlstm",end='==')
if args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
    print('convgru1',end='==')
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
    # encoder_params = convlstm_encoder_params
    # decoder_params = convlstm_decoder_params


 

def train():
    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)

    
    # 尝试add grap===============================
    # for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
    #     inputs = inputVar.to("cuda:0")  # B,S,C,H,W
    #     tb.add_graph(net, inputs)
    #     #Too complicated graph for ConvLSTM.......
    #     # vis_graph = make_dot(net(inputs), params=dict(net.named_parameters()))
    #     # vis_graph.view()    
    #     print("add graph succeed")
    #     break
    # #=========================================

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=5, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device) 
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    #check point断点续传
    if os.path.exists(os.path.join(save_dir, 'checkpoint_8_0.002563.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint_8_0.002563.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)  #half the learning rate every 4 epoch without improving. No print message.



    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf

    
    scaler = GradScaler()   


    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        print("Now train=====================",end='')
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            optimizer.zero_grad()
            net.train()
            with autocast():
                pred = net(inputs)  # B,S,C,H,W
                loss = lossfunction(pred, label)
            
            # add images of current epoch

            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            # loss.backward()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0) 

            scaler.step(optimizer)
            scaler.update()

            #  防止梯度爆炸：parameters (Iterable[Tensor] or Tensor) – an iterable of Tensors or a single Tensor that will have gradients normalized
            # clip_value (float or int) – maximum allowed value of the gradients. 


            # optimizer.step()
            t.set_postfix({ #进度条上显示数值
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })

        image_epoch=create_video(inputs,pred,label)
        tb.add_image('Epoch_Train' + str(cur_epoch), image_epoch)
        print('adding train images!'+str(cur_epoch))
        tb.close()  
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        tb.close()
        ######################
        # validate the model #
        ######################

        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            print("Then Test=====================",end="")
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                with autocast():
                    pred = net(inputs)
                    loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                # print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })
                # add image

            # add image of this epoch
        image_epoch=create_video(inputs,pred,label)
        tb.add_image('Epoch_Test' + str(cur_epoch), image_epoch)
        print('adding test images!')
        tb.close()  

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        tb.close()
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open(TIMESTAMP+"avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open(TIMESTAMP+"avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train()
