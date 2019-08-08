import torch
from torch.nn import functional as F  # Function-style
import torchvision.models as pre_models
import torch.utils.model_zoo as model_zoo

vgg16_no_bn_url= 'https://download.pytorch.org/models/vgg16-397923af.pth'
vgg19_bn_url='https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'

class Reshape(torch.nn.Module):
    def __init__(self, c, h, w):
        super(Reshape, self).__init__()
        self.c = c
        self.h = h
        self.w = w

    def forward(self, x):  # x is input
        return x.view(x.size(0), self.c, self.h, self.w)


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):  # x is input
        return x.view(x.size(0), -1)


class ResBlk(torch.nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        '''
            two conv:
            b*ch_in*h*w  =>
            b*ch_out*h*w (可能下采样)  =>
            b*ch_out*h*w
        '''
        super(ResBlk, self).__init__()
        self.left = torch.nn.Sequential(
            torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.InstanceNorm2d(ch_out, affine=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.InstanceNorm2d(ch_out, affine=True),
        )
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or ch_out != ch_in:  # change input channels to ch_out or kernal size using 1*1 Conv  (diff from original version)
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False),
                torch.nn.InstanceNorm2d(ch_out, affine=True),
            )

    def forward(self, x):
        return F.relu(self.left(x)+self.shortcut(x), inplace=True)


class Generator(torch.nn.Module):
    def __init__(self, input_height, input_width):
        super(Generator, self).__init__()
        self.resblock_inchannel = 256
        self.ih = input_height
        self.iw = input_width

        self.conv1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # torch.nn.InstanceNorm2d(64, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.conv1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # torch.nn.InstanceNorm2d(64, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # torch.nn.InstanceNorm2d(128, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # torch.nn.InstanceNorm2d(128, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # torch.nn.InstanceNorm2d(256, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # torch.nn.InstanceNorm2d(256, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3_3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # torch.nn.InstanceNorm2d(256, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.pool_3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # torch.nn.InstanceNorm2d(512, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.conv4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # torch.nn.InstanceNorm2d(512, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.conv4_3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # torch.nn.InstanceNorm2d(512, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.pool_4 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # ks取2 padding咋整啊

        self.conv5_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),  # atrous convolutional layer
            # torch.nn.InstanceNorm2d(512, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.conv5_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),  # atrous convolutional layer
            # torch.nn.InstanceNorm2d(512, affine=True),
            torch.nn.ReLU(inplace=True),
        )
        self.conv5_3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),  # atrous convolutional layer
            # torch.nn.InstanceNorm2d(512, affine=True),
            torch.nn.ReLU(inplace=True),
        )

        self.conv_layers_for_upsample = torch.nn.ModuleList([])
        self.in_channels = [64]*2 + [128]*2 + [256]*3 + [512]*6
        self.out_channels = [16, 16, 16, 16] + [32]*9
        for i in range(len(self.in_channels)):
            self.conv_layers_for_upsample.append(torch.nn.Conv2d(self.in_channels[i], self.out_channels[i],
                                                                 kernel_size=3, stride=1, padding=1))  # bilinear interpolation init?

        # bilinear interpolation init?
        self.deconv1 = torch.nn.Sequential(
            torch.nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),  # 192 = 32*6
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=2-1),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        self.deconv2 = torch.nn.Sequential(
            torch.nn.Conv2d(352, 128, kernel_size=3, stride=1, padding=1),  # 352 = 256 + 32*3
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=2-1),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        self.deconv3 = torch.nn.Sequential(
            torch.nn.Conv2d(160, 64, kernel_size=3, stride=1, padding=1),  # 160 = 128 + 16*2
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=2-1),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        self.last_conv = torch.nn.Sequential(
            torch.nn.Conv2d(96, 16, kernel_size=3, stride=1, padding=1),  # 96 = 64 + 16*2
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid(),
        )

    # def make_layer(self, block, ch_out, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)  # strides=[1,1] or [2,1]
    #     layers = []
    #     for one_stride in strides:
    #         layers.append(block(self.resblock_inchannel, ch_out, one_stride))
    #         self.resblock_inchannel = ch_out
    #     return torch.nn.Sequential(*layers)


    def update_pre_train_para_with_VGG16(self, num_layers = 26):
        pretrained_dict = model_zoo.load_url(vgg16_no_bn_url)
        pretrained_dict_list = list(pretrained_dict.keys())
        model_dict = self.state_dict()
        model_dict_list = list(model_dict)
        for i in range(0, num_layers):  # 26 you try , if wrong , let me know , thanks
            assert model_dict[model_dict_list[i]].shape == pretrained_dict[pretrained_dict_list[i]].shape
            model_dict[model_dict_list[i]] = pretrained_dict[pretrained_dict_list[i]]


    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        pool_1 = self.pool_1(conv1_2)
        conv2_1 = self.conv2_1(pool_1)
        conv2_2 = self.conv2_2(conv2_1)
        pool_2 = self.pool_2(conv2_2)
        conv3_1 = self.conv3_1(pool_2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        pool_3 = self.pool_3(conv3_3)
        conv4_1 = self.conv4_1(pool_3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        pool_4 = self.pool_4(conv4_3)
        conv5_1 = self.conv5_1(pool_4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)

        conv4_1 = F.relu(self.conv_layers_for_upsample[7](conv4_1), inplace=True)
        conv4_2 = F.relu(self.conv_layers_for_upsample[8](conv4_2), inplace=True)
        conv4_3 = F.relu(self.conv_layers_for_upsample[9](conv4_3), inplace=True)
        conv5_1 = F.relu(self.conv_layers_for_upsample[10](conv5_1), inplace=True)
        conv5_2 = F.relu(self.conv_layers_for_upsample[11](conv5_2), inplace=True)
        conv5_3 = F.relu(self.conv_layers_for_upsample[12](conv5_3), inplace=True)
        Up_1 = self.deconv1(torch.cat([conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3], dim=1))  # ouput_channel: 256

        conv3_1 = F.relu(self.conv_layers_for_upsample[4](conv3_1), inplace=True)
        conv3_2 = F.relu(self.conv_layers_for_upsample[5](conv3_2), inplace=True)
        conv3_3 = F.relu(self.conv_layers_for_upsample[6](conv3_3), inplace=True)
        Up_2 = self.deconv2(torch.cat([Up_1, conv3_1, conv3_2, conv3_3], dim=1))  # ouput_channel: 128

        conv2_1 = F.relu(self.conv_layers_for_upsample[2](conv2_1), inplace=True)
        conv2_2 = F.relu(self.conv_layers_for_upsample[3](conv2_2), inplace=True)
        Up_3 = self.deconv3(torch.cat([Up_2, conv2_1, conv2_2], dim=1))  # ouput_channel: 64

        conv1_1 = F.relu(self.conv_layers_for_upsample[0](conv1_1), inplace=True)
        conv1_2 = F.relu(self.conv_layers_for_upsample[1](conv1_2), inplace=True)
        Contour = self.last_conv(torch.cat([Up_3, conv1_1, conv1_2], dim=1))  # ouput_channel: 1

        return Contour


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),  # output: bs*64*h/4*w/4
            torch.nn.InstanceNorm2d(64, affine=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # output: bs*128*h/8*w/8
            torch.nn.InstanceNorm2d(128, affine=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),  # output: bs*128*h/16*w/16
            torch.nn.InstanceNorm2d(128, affine=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # output: bs*256*13*13
            torch.nn.InstanceNorm2d(256, affine=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # output: bs*256*7*7
            torch.nn.InstanceNorm2d(256, affine=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # output: bs*512*4*4
            torch.nn.InstanceNorm2d(512, affine=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # output: bs*512*2*2
            torch.nn.InstanceNorm2d(512, affine=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            Flatten(),
            torch.nn.Linear(2048, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x
