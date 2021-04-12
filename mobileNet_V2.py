import torchvision.models as pretrained_model
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class mobileNetV2(nn.Module):
    def __init__(
        self,
        arch_backbone="mobilenet_v2",
        CNN_embed_dim=256,
        drop_p=0.3,
        num_classes=10,
    ):
        super(mobileNetV2, self).__init__()

        self._define_layers()
        self.drop_p = drop_p
        self.CNN_embed_dim = CNN_embed_dim
        self.output_static_layers = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=CNN_embed_dim, out_features=num_classes, bias=True),
        )

    def _define_layers(self):
        self.stage1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.PReLU(32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 64, stride=2),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            BlazeBlock(64, 128, stride=2),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            BlazeBlock(128, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )
        self.stage4 = nn.Sequential(
            BlazeBlock(256, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            # resize_bilinear
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=1, padding=0, bias=True
            ),
            nn.PReLU(256),
        )
        self.stage5 = nn.Sequential(
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )
        self.stage6 = nn.Sequential(
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
        )

        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.PReLU(128),
        )

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        bs, T, C, H, W = x.size()
        # cnn_embed_seq = []
        x = x.view(-1, C, H, W)

        x = F.pad(x, (1, 2, 1, 2), "constant", 0)

        x = self.stage1(x)  # (b, 32, 128, 128)
        x = self.stage2(x)  # (b, 64, 64, 64)
        h_1 = x

        x_s3 = self.stage3(x)  # (b, 128, 32, 32)
        x = self.stage4(x_s3) + x_s3
        x_s5 = self.stage5(x)

        x_d1 = h_1 + self.deconv1(x_s5)
        x_s6 = self.stage6(x_d1)
        x_s6 = x_s6.reshape(bs * T, -1)
        in_feats = x_s6.shape[1]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_feats, out_features=self.CNN_embed_dim, bias=True),
        )
        x_2d = self.classifier(x_s6)
        print(x_2d.shape)
        output_static = self.output_static_layers(x_2d)
        print(output_static.shape)
        cnn_embed_seq = x_2d.view(bs, T, -1)
        print(cnn_embed_seq.shape)

        return cnn_embed_seq, output_static


if __name__ == "__main__":
    model = mobileNetV2()
    _input = torch.randn(4, 9, 3, 224, 224)
    output = model(_input)
    print("output shape:", output.shape)