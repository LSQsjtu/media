import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import T


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, t=6):
        super(BlazeBlock, self).__init__()

        self.channel_pad = out_channels - in_channels

        padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * t,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU6(),
            #batchnorm
            nn.Conv2d(
                in_channels=in_channels * t,
                out_channels=in_channels * t,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=in_channels * t,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x):
        # if self.stride == 2:
        #     h = F.pad(x, (1, 2, 1, 2), "constant", 0)
        # else:
        h = x

        # if self.channel_pad > 0:
        #     x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.convs(h) + x


class cnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, t=6, stride=1):
        super(cnnBlock, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * t,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU6(),
        )
        self.convs2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels * t,
                out_channels=in_channels * t,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=in_channels * t,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x):
        self.convs1(x)
        if self.stride == 2:
            if self.kernel_size == 5:
                x = F.pad(x, (1, 2, 1, 2), "constant", 0)
            else:
                x = F.pad(x, (0, 2, 0, 2), "constant", 0)

        return self.convs2(x)


class BlazeHand(nn.Module):
    """The BlazeFace face detection model from MediaPipe.

    The version from MediaPipe is simpler than the one in the paper;
    it does not use the "double" BlazeBlocks.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """

    def __init__(self):
        super(BlazeHand, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        self.score_clipping_thresh = 100.0
        self.min_score_thresh = 0.75

        self._define_layers()

    def _define_layers(self):
        self.stage1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=24,
                out_channels=24,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=24,
                bias=True,
            ),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=24,
                out_channels=8,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            cnnBlock(8, 16, kernel_size=3, t=4, stride=2),
            BlazeBlock(16, 16, kernel_size=3, t=4),
            cnnBlock(16, 24, kernel_size=5, t=4, stride=2),
            BlazeBlock(24, 24, kernel_size=5, t=6),
            cnnBlock(24, 40, kernel_size=3, t=6, stride=2),
            BlazeBlock(40, 40, kernel_size=3, t=6),
            BlazeBlock(40, 40, kernel_size=3, t=6),
            cnnBlock(40, 56, kernel_size=5, t=6, stride=1),
            BlazeBlock(56, 56, kernel_size=5, t=6),
            BlazeBlock(56, 56, kernel_size=5, t=6),
            cnnBlock(56, 96, kernel_size=5, t=6, stride=2),
            BlazeBlock(96, 96, kernel_size=5, t=6),
            BlazeBlock(96, 96, kernel_size=5, t=6),
            BlazeBlock(96, 96, kernel_size=5, t=6),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=576,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=576,
                out_channels=576,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=576,
                bias=True,
            ),
            nn.ReLU6(),
        # )
        # self.stage3 = nn.Sequential(
            nn.Conv2d(
                in_channels=576,
                out_channels=576,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=576,
                bias=True,
            ),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=576,
                out_channels=144,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        self.fc1 = nn.Linear(2304, 1)
        self.fc2 = nn.Linear(2304, 1)
        self.fc3 = nn.Linear(2304, 63)

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)

        b = x.shape[0]  # batch size, needed for reshaping later

        x = self.stage1(x)  # (b, 96, 7, 7)
        print("stage1 shape: ", x.shape)
        x = self.stage2(x)  # (b, 64, 64, 64)
        print("stage2 shape: ", x.shape)
        x.reshape(b, -1)

        # x = self.stage3(x)           # (b, 128, 32, 32)
        # print('stage3 shape: ', x.shape)
        # x = self.stage4(x)           # (b, 192, 16, 16)
        # print('stage4 shape: ', x.shape)
        # x = self.stage5(x)           # (b, 192, 8, 8)
        # print('stage5 shape: ', x.shape)
        # x = self.stage6(x)           # (b, 192, 4, 4)
        # print('stage6 shape: ', x.shape)
        # x = self.stage7(x)           # (b, 192, 2, 2)
        # print('stage7 shape: ', x.shape)

        hand_flag = self.fc1(x)
        hand_flag = torch.sigmoid(hand_flag)
        hand_flag = hand_flag.view(b, 1)

        handness = self.fc2(x)
        handness = torch.sigmoid(handness)
        handness = handness.view(b, 1)

        hand_coords = self.fc3(x)
        hand_coords = hand_coords.view(b, -1)

        return [hand_flag, handness, hand_coords]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0
