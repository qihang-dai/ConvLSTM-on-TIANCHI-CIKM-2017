import numpy as np
import os
from tqdm import tqdm

def read_data(input_file,pic_ind,T_ind,H_ind):
    TH_ind = (T_ind-1)*4 + (H_ind - 1)
    f = open(input_file, "r")
    f.seek( ((pic_ind -1)*60 + TH_ind)*101*101, os.SEEK_SET)  # seek #从头找到此位置的数据
    data = np.fromfile(  f, count = 101*101, dtype = np.ubyte)
    f.close()
    data_mat = data.reshape(101,101)
    return data_mat





inputfile=r"D:\download\tianchi\data\train_ubyte.txt"
 # time index: 1 - 15
H_ind = 3    # height index:  1 - 4 (0.5km - 3.5 km)
# image index: 1-10000 in train, 1- 2000 in testA
# testimg=read_data(input,Img_ind,T_ind,H_ind)

Iarray = np.zeros((10000,14,100,100)).astype(np.ubyte)
for Img_ind in tqdm(range(1,10001)):
    Tarray = np.zeros((14,100,100)).astype(np.ubyte)
    for T_ind in range(1,15):
        data_mat = read_data(inputfile,Img_ind,T_ind,H_ind)
        data_mat2= data_mat[1:,1:]
        # print(data_mat2.shape)
        # print("mat readed",end=' ',flush=True)
        Tdata=data_mat2[np.newaxis,:]
        Tarray[T_ind-1,:]=Tdata
        # print(res[T_id-1,:].shape)
    # print(Tarray.shape)
    Idata=Tarray[np.newaxis,:]
    Iarray[Img_ind-1,:]=Idata
    
Iarray=Iarray.swapaxes(0,1)
dataset=Iarray[:,:,:,:,np.newaxis]     
  
print(dataset.shape)
print((dataset[0,:,:,:,:]==dataset[1,:,:,:,:]).all())
np.save(r"D:\download\ConvLSTMpytorch\Radar\data\train.npy", dataset)

