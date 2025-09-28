import torch
import torch.nn as nn
from torchvision import models
from einops import rearrange
from torchvision.models._utils import IntermediateLayerGetter

class VGG_16(nn.Module):
    def __init__(self, hidden, ss = [2,2,2,2,2], ks=[(3,3),(3,3),(3,3),(3,3),(3,3)]):
        super(VGG_16,self).__init__()
        self.relu = nn.ReLU()
        self.block1_conv1 = nn.Conv2d(3,64,(3,3),1,padding='constant')
        self.block1_conv2 = nn.Conv2d(64,64,(3,3),1,padding='constant')

        self.block1_pool = nn.AvgPool2d(ks[0],ss[0])

        self.block2_conv1 = nn.Conv2d(64,128,(3,3),1,padding='constant')
        self.block2_conv1 = nn.Conv2d(128,128,(3,3),1,padding='constant')

        self.block2_pool = nn.AvgPool2d(ks[1],ss[1])

        self.block3_conv1 = nn.Conv2d(128,256,(3,3),1,padding='constant')
        self.block3_conv2 = nn.Conv2d(256,256,(3,3),1,padding='constant')
        self.block3_conv3 = nn.Conv2d(256,256,(3,3),1,padding='constant')

        self.block3_pool = nn.AvgPool2d(ks[2],ss[2])

        self.block4_conv1 = nn.Conv2d(256,512,(3,3),1,padding='constant')
        self.block4_conv2 = nn.Conv2d(512,512,(3,3),1,padding='constant')
        self.block4_conv3 = nn.Conv2d(512,512,(3,3),1,padding='constant')

        self.block4_pool = nn.AvgPool2d(ks[3],ss[3])

        self.block5_conv1 = nn.Conv2d(512,512,(3,3),1,padding='constant')
        self.block5_conv2 = nn.Conv2d(512,512,(3,3),1,padding='constant')
        self.block5_conv3 = nn.Conv2d(512,512,(3,3),1,padding='constant')

        self.block5_pool = nn.AvgPool2d(ks[4],ss[4])
        self.dropout = nn.Dropout(0.1)
        self.last_conv = nn.Conv2d(512,hidden,1)
    
    def forward(self,x):
        x = self.relu(self.block1_conv1(x))
        x = self.relu(self.block1_conv2(x))

        x = self.block1_pool(x)

        x = self.relu(self.block1_conv1(x))
        x = self.relu(self.block1_conv1(x))

        x = self.block2_pool(x)

        x = self.relu(self.block1_conv1(x))
        x = self.relu(self.block1_conv1(x))
        x = self.relu(self.block1_conv1(x))

        x = self.block3_pool(x)

        x = self.relu(self.block1_conv1(x))
        x = self.relu(self.block1_conv1(x))
        x = self.relu(self.block1_conv1(x))

        x = self.block4_pool(x)

        x = self.relu(self.block1_conv1(x))
        x = self.relu(self.block1_conv1(x))
        x = self.relu(self.block1_conv1(x))

        x = self.block5_pool(x)

        x = self.dropout(x)

        x = self.last_conv(x)

        x = x.transpose(-1, -2)
        x = x.flatten(2)
        x = x.permute(-1,0,1)

        return x
    
class Vgg(nn.Module):
    def __init__(self, ss, ks, hidden, pretrained=True, dropout=0.1):
        super(Vgg, self).__init__()

        if pretrained:
            weights = "DEFAULT"
        else:
            weights = None

        cnn = models.vgg19_bn(weights=weights)

        pool_idx = 0

        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn.features[i] = torch.nn.AvgPool2d(
                    kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0
                )
                pool_idx += 1

        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

    def forward(self, x):
        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        #        conv = rearrange(conv, 'b d h w -> b d (w h)')
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv