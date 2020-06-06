import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

torch.autograd.set_detect_anomaly(True)
from .ozan_rep_fun import ozan_rep_function,trevor_rep_function,OzanRepFunction,TrevorRepFunction

__all__ = ['a2d2net_taskonomy']

class BasicBlock(nn.Module):
  def __init__(self, inchannels, outchannels, kernel, padding, pool=2):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=inchannels,out_channels=inchannels,kernel_size=kernel,padding=padding)
    self.bn1 = nn.BatchNorm2d(inchannels)
    self.conv2 = nn.Conv2d(in_channels=inchannels,out_channels=inchannels,kernel_size=kernel,padding=padding)
    self.bn2 = nn.BatchNorm2d(inchannels)

    self.conv_inp = nn.Conv2d(in_channels=inchannels,out_channels=outchannels,kernel_size=kernel, padding=padding)
    self.maxpool = nn.MaxPool2d(pool)

  def forward(self,inpt):

    x = self.conv1(inpt)
    x = self.bn1(x)
    x = nn.ReLU()(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = nn.ReLU()(x)

    x += inpt
    x = self.maxpool(x)
    x = self.conv_inp(x)

    return x

class A2D2Encoder(nn.Module):
    # Input images are of size 256x256.
    def __init__(self):
        super(A2D2Encoder, self).__init__()

        self.final_channels = 64
        self.num_blocks = 4
        self.hw_final = 256 // (2**(self.num_blocks+1))

        self.conv1 = nn.Conv2d(3,3,kernel_size= 5,padding = 2)
        self.conv2 = nn.Conv2d(3,8,kernel_size= 5,padding = 2)
        self.block1 = BasicBlock(8,16,5,2)
        self.block2 = BasicBlock(16,32,5,2)
        self.block3 = BasicBlock(32,64,3,1)
        self.block4 = BasicBlock(64,self.final_channels,3,1,4)
        # Image size: 256 -> 128 -> 64 -> 32 -> 8 : 256//(2**5)


        self.fc1 = nn.Linear(self.final_channels*8*8, 3*8*8)
        self.fc2 = nn.Linear(3*8*8, 3*8*8)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.view(-1,self.final_channels*8*8)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = x.view(-1,3,8,8)

        return x



class UpConv(nn.Module):
  def __init__(self,inchannels, outchannels):
    super(UpConv,self).__init__()
    # stride is always 2
    self.upconv = nn.ConvTranspose2d(inchannels,outchannels,2,2) # kernel of sie 2 is apt.
    self.bn_upconv = nn.BatchNorm2d(outchannels)
    self.conv = nn.Conv2d(outchannels,outchannels,3,padding=1)
    self.bn_conv = nn.BatchNorm2d(outchannels)
    self.final_conv = nn.Conv2d(outchannels,outchannels,1)

  def forward(self,x):
    x = self.upconv(x)
    x = self.bn_upconv(x)
    x = nn.ReLU()(x) 
    x = self.conv(x)
    x = self.bn_conv(x)
    x = nn.ReLU()(x)
    x = self.final_conv(x)
    return x


class A2D2Decoder(nn.Module):
    # output of encoder is 3*8*8
    def __init__(self, out_channels):
        super(A2D2Decoder, self).__init__()
        self.upconv1 = UpConv(3,8)
        self.upconv2 = UpConv(8,16)
        self.upconv3 = UpConv(16,32)
        self.upconv4 = UpConv(32,64)
        self.upconv5 = UpConv(64,out_channels)
        # Size: 8 -> 16 -> 32 -> 64 -> 128 -> 256


    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        return x


class A2D2Net(nn.Module):
    def __init__(self, tasks=None):
        super(A2D2Net, self).__init__()

        self.encoder = A2D2Encoder()
        
        self.tasks=tasks
        self.task_to_decoder = {}

        if tasks is not None:
            for task in tasks:
                if task == 'segment_semantic':
                    output_channels = 18
                if task == 'depth_zbuffer':
                    output_channels = 1
                if task == 'normal':
                    output_channels = 3
                if task == 'edge_occlusion':
                    output_channels = 1
                if task == 'reshading':
                    output_channels = 3
                if task == 'keypoints2d':
                    output_channels = 1
                if task == 'edge_texture':
                    output_channels = 1

                decoder = A2D2Decoder(output_channels)
                self.task_to_decoder[task]=decoder
        
        self.decoders = nn.ModuleList(self.task_to_decoder.values())

    def forward(self, input):

        rep = {task: self.encoder(im) for task, im in input.items()}
        outputs={}

        # rep = ozan_rep_function(rep)
        # rep = trevor_rep_function(rep)

        for (task,decoder) in zip(self.task_to_decoder.keys(), self.decoders):
            outputs[task] = decoder( trevor_rep_function( rep[task] ) )
        
        return outputs


def a2d2net_taskonomy(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _tinynet('tinynet', Bottleneck, [3, 8, 36, 3], pretrained,
    #                **kwargs)
    return A2D2Net(**kwargs)
