import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, padding=1, bn_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, bias=not bn_act)
        self.norm = nn.BatchNorm2d(out_chan) if bn_act else nn.Identity()
        self.act = nn.ReLU() if bn_act else nn.Identity()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class PredBlock(nn.Module):
    def __init__(self, in_chan, anchors, num_class):
        super().__init__()
        self.num_class = num_class
        self.conv = nn.Sequential(
            CNNBlock(in_chan, 2*in_chan, 3, 1, 1, bn_act=True),
            nn.Conv2d(2*in_chan, anchors * (5 + num_class), kernel_size=1, stride=1, padding=0)
        )
        self.anchors_per_scale = anchors
    def forward(self, x):
        b, _, h, w = x.shape
        x = self.conv(x)
        # (b, anchors*(5+num_class), h, w) -> (b, anchors, 5+num_class, h, w) -> (b, anchors, h, w, 5+num_class)
        x = x.view(b, self.anchors_per_scale, 5 + self.num_class, h, w).permute(0, 1, 3, 4, 2).contiguous()
        return x  # (B, A, H, W, 5+num_class)

class ResidualBlock(nn.Module):
    def __init__(self, channels, repeats=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                CNNBlock(channels, channels//2, 1, 1, 0, True),
                CNNBlock(channels//2, channels, 3, 1, 1, True)
            ) for _ in range(repeats)
        ])
    def forward(self, x):
        for m in self.layers:
            x = x + m(x)
        return x

class MiniYolo(nn.Module):
    def __init__(self, in_chan=3, num_class=10, anchors=3):
        super().__init__()
        self.backbone = nn.Sequential(
            CNNBlock(in_chan, 64, 3, 1),
            CNNBlock(64, 128, 3, 1),
            nn.MaxPool2d(2,2),
            
            CNNBlock(128, 256, 3, 1),
            CNNBlock(256, 512, 3, 1),
            nn.MaxPool2d(2,2),

            CNNBlock(512, 512, 3, 1),
            CNNBlock(512, 512, 3, 1),
            nn.MaxPool2d(2,2),


        )
        self.head = PredBlock(256, anchors=anchors, num_class=num_class)
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
    

class MiniYoloMultiScale(nn.Module):
    def __init__(self, in_chan=1, num_class=10, anchors=5):
        super().__init__()
        self.conv1 = CNNBlock(in_chan, 64, 3, 1)
        self.conv2 = CNNBlock(64, 128, 3, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.deep1 = ResidualBlock(128,2)
        self.conv3 = CNNBlock(128, 256, 3, 1)
        self.conv4 = CNNBlock(256, 512, 3, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.deep2 = ResidualBlock(512,2)
        self.conv5 = CNNBlock(512, 512, 3, 1)
        self.conv6 = CNNBlock(512, 512, 3, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.deep3 = ResidualBlock(512,2)
        self.conv7 = CNNBlock(512, 512, 3, 1)
        self.conv8 = CNNBlock(512, 256, 3, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.pred_block_0 = PredBlock(128, anchors=anchors, num_class=num_class)
        self.pred_block_1 = PredBlock(128, anchors=anchors, num_class=num_class)
        self.pred_block_2 = PredBlock(512, anchors=anchors, num_class=num_class)
        self.pred_block_3 = PredBlock(512, anchors=anchors, num_class=num_class)
        self.pred_block_4 = PredBlock(256, anchors=anchors, num_class=num_class)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        feature_0 = x
        x = self.pool1(x)
        x = self.deep1(x)
        feature_1 = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.deep2(x)
        feature_2 = x
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.deep3(x)
        feature_3 = x
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool4(x)
        feature_4 = x
        pred_0 = self.pred_block_0(feature_0)
        pred_1 = self.pred_block_1(feature_1)
        pred_2 = self.pred_block_2(feature_2)
        pred_3 = self.pred_block_3(feature_3)
        pred_4 = self.pred_block_4(feature_4)
        return pred_0, pred_1, pred_2, pred_3, pred_4