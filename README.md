# Easy-Use-ConvLSTM and ConvGRU-for-beginners
## **Who are supposed to read this repo** 

A rookie modified ConvLSTM source code.

Rookies in deep learning who mistakenly plan to use ConvLSTM for final year project (just like me) or something. Anyone familiared with CNN and LSTM will find this repo not worthy of reading.

This repo provide a shortpath on how to use the current source code on your dataset. There are some redundant codes.

![image](https://user-images.githubusercontent.com/13762187/113415518-f6af0a00-93f1-11eb-8a3b-3c0eee667f2b.png)


## Other repo
Majority of code comes from https://github.com/jhhuang96/ConvLSTM-PyTorch.

https://github.com/holmdk/Video-Prediction-using-PyTorch also deploy ConvLSTM by Pytorch-lighting. The codes of models are much more concise and understandable for beginners.

# How to use this repo on your dataset

You need to modify the Decoder and Encoder's parameter, depends on your original matrix widths and lengths. You need to do some calculation to decide the size of each CNN layers's kernel.

## Open source dataset
Rainnet: in ndf format:

Tianchi 2017

HKO(student need their advisor to send application)

## Step.1 Data preprocess

## Unsolved Issues

Tensorboard only add_image the first epoch of image in one run.
