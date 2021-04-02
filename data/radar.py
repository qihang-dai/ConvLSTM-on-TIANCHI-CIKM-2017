import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data



def load_fixed_set(root, is_train):
    # Load the fixed dataset
    if is_train==False:
        filename = 'testA_100.npy'
    elif is_train==True:
        filename = 'train.npy'
    else:
        print('Please choose is_train ture or False')

    path = os.path.join(root, filename)
    dataset = np.load(path)
    return dataset


class Radar(data.Dataset):
    def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
                 transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(Radar, self).__init__()
        self.dataset = load_fixed_set(root, is_train)
        self.length = self.dataset.shape[1]
        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform



    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output #20
        images = self.dataset[:, idx, ...] # [20,64,64,1]   #（14，100，100，1）
        
        images = images[:,:,:,0]    #（14，100，100）
        images=images[:,np.newaxis,:,:] #（14，1，100，100）
 
        input = images[:self.n_frames_input] #10，1，64，64

        output = images[self.n_frames_input:length]


        frozen = input[-1]
        # add a wall to input data
        # pad = np.zeros_like(input[:, 0])
        # pad[:, 0] = 1
        # pad[:, pad.shape[1] - 1] = 1
        # pad[:, :, 0] = 1
        # pad[:, :, pad.shape[2] - 1] = 1
        #
        # input = np.concatenate((input, np.expand_dims(pad, 1)), 1)

        output = torch.from_numpy(output / 255.0).contiguous().float() #除以255？Normalize into 0-1
        input = torch.from_numpy(input / 255.0).contiguous().float()
        # print()
        # print(input.size())
        # print(output.size())

        out = [idx, output, input, frozen, np.zeros(1)] 
        return out

    def __len__(self):
        return self.length

if __name__ == "__main__":
    trainFolder = Radar(is_train=False,
                          root='data/',
                          n_frames_input=7,
                          n_frames_output=7,
                          num_objects=[2])
    trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=4,
                                          shuffle=False)
    #   #S   B    OUTPUT      INPUT  FORZEN  0
    for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
            inputs = inputVar # B,S,1,64,64
            print("runing")
            break
    print("inputs.shape",inputs.shape)
    print("inputs[0].shape",inputs[0].shape)  # S，1，H，W       Aim: 3S,1,H,W
    print("inputs[0,0].shape",inputs[0,0].shape)