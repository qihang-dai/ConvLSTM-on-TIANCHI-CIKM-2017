from torch import nn
from collections import OrderedDict
import torch
import torchvision
import argparse

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))

def create_video(x, y_hat, y):#x=input_first 10 frames,y_hat=x.predict, y= input-later 10 frames(ground truth)
        # predictions with input for illustration purposes
        preds = torch.cat([x, y_hat], dim=1)[0] #BS1HW TO S1WH. First row: input+output

        # entire input and ground truth
        y_plot = torch.cat([x, y], dim=1)[0] #同上 S1WH

        # error (l2 norm) plot between pred and ground truth
        difference = (torch.pow(y_hat[0].squeeze() - y[0].squeeze(), 3)).detach() #How to change from Y_hat BS1HW to SHW?: [0] make S1HW,
        zeros = torch.zeros(difference.shape).cuda()
     
        difference_plot = torch.cat([zeros.unsqueeze(0), difference.unsqueeze(0)], dim=1)[0].unsqueeze(1)  
        #difference plot's shape become S 1 H W <--[Step.unsqueeze(1)] S H W  <--[Step.[0]] 1 S H W  <--[Step.cat.dim 1] 1 S/2 H W <--[Step.unsqueeze(0)] S/2 H W 
        #Thus zero=3D tensor SHW

        # concat all images
        final_image = torch.cat([preds, y_plot, difference_plot], dim=0)
        #final iamge shape is 3*S 1 HW
        # make them into a single grid image file
        grid = torchvision.utils.make_grid(final_image, nrow=14)
        #make_grid的作用是将若干幅图像拼成一幅图像。其中padding的作用就是子图像与子图像之间的pad有多宽
        return grid 
