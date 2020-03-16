from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
from layers.l2norm import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
    """
    VGG base convolutinos to produce lower-level feature maps
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolution layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)   # 此处和SSD不一样

         # Replacements for FC6 and FC7 in VGG16
        self.conv_fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv_fc7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.load_pretrained_layers()

    def forward(self, image):
        """
        :param image: a tensor of (N, 3, 640, 640)
        :return: lower-level feature maps 
        """

        out = F.relu(self.conv1_1(image))  # (N, 64, 640, 640)
        out = F.relu(self.conv1_2(out))  # (N, 64, 640, 640)
        out = self.pool1(out)  # (N, 64, 320, 320)

        out = F.relu(self.conv2_1(out))  # (N, 128, 320, 320)
        out = F.relu(self.conv2_2(out))  # (N, 128, 320, 320)
        out = self.pool2(out)  # (N, 128, 160, 160)

        out = F.relu(self.conv3_1(out))  # (N, 256, 160, 160)
        out = F.relu(self.conv3_2(out))  # (N, 256, 160, 160)
        out = F.relu(self.conv3_3(out))  # (N, 256, 160, 160)
        conv3_3_feats = out
        out = self.pool3(out)  # (N, 256, 80, 80), it would have been 37 if not for ceil_mode = True

        out = F.relu(self.conv4_1(out))  # (N, 512, 80, 80)
        out = F.relu(self.conv4_2(out))  # (N, 512, 80, 80)
        out = F.relu(self.conv4_3(out))  # (N, 512, 80, 80)
        conv4_3_feats = out  # (N, 512, 80, 80)
        out = self.pool4(out)  # (N, 512, 40, 40)
    
        out = F.relu(self.conv5_1(out))  # (N, 512, 40, 40)
        out = F.relu(self.conv5_2(out))  # (N, 512, 40, 40)
        out = F.relu(self.conv5_3(out))  # (N, 512, 40, 40)
        conv5_3_feats = out
        out = self.pool5(out)   # (N, 512, 20, 20)

        out = F.relu(self.conv_fc6(out))  # (N, 1024, 20, 20)

        conv_fc7_feats = F.relu(self.conv_fc7(out))  # (N, 1024, 20, 20)

        # Lower-level feature maps
        # conv3_3_feats, conv4_3_feats, conv5_3_feats 会再经过一个Norm2D层
        return conv3_3_feats, conv4_3_feats, conv5_3_feats, conv_fc7_feats


    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv_fc6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv_fc6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv_fc7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv_fc7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

                # Auxiliary/additional convolutions on top of the VGG base
        # 对应S3FD论文中的conv6_1和conv6_2
        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        # 对应S3FD论文中的conv7_1和conv7_2
        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    
    def forward(self, conv_fc7_feats):
        """
        Forward propagation.

        :param conv_fc7_feats: lower-level conv_fc7_feats feature map, a tensor of dimensions (N, 1024, 20, 20)
        :return: higher-level feature maps conv6_2, conv7_2
        """

        out = F.relu(self.conv6_1(conv_fc7_feats))  # (N, 256, 20, 20)
        out = F.relu(self.conv6_2(out))  # (N, 512, 10, 10)
        conv6_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv7_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv7_2(out))  # (N, 256, 5, 5)
        conv7_2_feats = out

        return conv6_2_feats, conv7_2_feats



class PredictionConvolutions(nn.Module):
