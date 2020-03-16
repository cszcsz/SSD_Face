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


    def __init__(self, n_classes):

        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes


        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv3_3 = nn.Conv2d(256, 4, kernel_size=3, padding=1)
        self.loc_conv4_3 = nn.Conv2d(512, 4, kernel_size=3, padding=1)
        self.loc_conv5_3 = nn.Conv2d(512, 4, kernel_size=3, padding=1)
        self.loc_conv_fc7 = nn.Conv2d(1024, 4, kernel_size=3, padding=1)
        self.loc_conv6_2 = nn.Conv2d(512, 4, kernel_size=3, padding=1)
        self.loc_conv7_2 = nn.Conv2d(256, 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        # conv3_3用了Max-out BG_label, 论文设置的Nm = 3， 所以Ns = Nm + n_class - 1 = 4
        self.cl_conv3_3 = nn.Conv2d(256, 3 + (self.n_classes - 1), kernel_size=3, padding=1)
        self.cl_conv4_3 = nn.Conv2d(512, self.n_classes, kernel_size=3, padding=1)
        self.cl_conv5_3 = nn.Conv2d(512, self.n_classes, kernel_size=3, padding=1)
        self.cl_conv_fc7 = nn.Conv2d(1024, self.n_classes, kernel_size=3, padding=1)
        self.cl_conv6_2 = nn.Conv2d(512, self.n_classes, kernel_size=3, padding=1)
        self.cl_conv7_2 = nn.Conv2d(256, self.n_classes, kernel_size=3, padding=1)

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

    def forward(self, conv3_3_feats, conv4_3_feats, conv5_3_feats, conv_fc7_feats, conv6_2_feats, conv7_2_feats):
        """
        Forward propagation.
        :param conv3_3_feats: conv3_3 feature map, a tensor of dimensions (N, 256, 160, 160)
        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 80, 80)
        :param conv5_3_feats: conv5_3 feature map, a tensor of dimensions (N, 512, 40, 40)
        :param conv_fc7_feats: conv_fc7_feats feature map, a tensor of dimensions (N, 1024, 20, 20)
        :param conv6_2_feats: conv6_2_feats feature map, a tensor of dimensions (N, 512, 10, 10)
        :param conv7_2_feats: conv7_2_feats feature map, a tensor of dimensions (N, 256, 5, 5)
        
        :return: 34125 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv3_3_feats.size(0)
        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        
        l_conv3_3 = self.loc_conv3_3(conv3_3_feats)  # (N, 4, 160, 160)
        l_conv3_3 = l_conv3_3.permute(0, 2, 3, 1).contiguous()  # (N, 160, 160, 4)，contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below
        l_conv3_3 = l_conv3_3.view(batch_size, -1, 4)   # (N, 25600, 4)

        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 4, 80, 80)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 80, 80, 4)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)   # (N, 6400, 4)

        l_conv5_3 = self.loc_conv5_3(conv5_3_feats)  # (N, 4, 40, 40)
        l_conv5_3 = l_conv5_3.permute(0, 2, 3, 1).contiguous()  # (N, 40, 40, 4)
        l_conv5_3 = l_conv5_3.view(batch_size, -1, 4)   # (N, 1600, 4)

        l_conv_fc7 = self.loc_conv_fc7(conv_fc7_feats)  # (N, 4, 20, 20)
        l_conv_fc7 = l_conv_fc7.permute(0, 2, 3, 1).contiguous()  # (N, 20, 20, 4)
        l_conv_fc7 = l_conv_fc7.view(batch_size, -1, 4)   # (N, 400, 4)

        l_conv6_2 = self.loc_conv6_2(conv6_2_feats)  # (N, 4, 10, 10)
        l_conv6_2 = l_conv6_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 4)
        l_conv6_2 = l_conv6_2.view(batch_size, -1, 4)   # (N, 100, 4)

        l_conv7_2 = self.loc_conv7_2(conv7_2_feats)  # (N, 4, 5, 5)
        l_conv7_2 = l_conv7_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 4)
        l_conv7_2 = l_conv7_2.view(batch_size, -1, 4)   # (N, 25, 4)

        # Predict classes in localization boxes
       
        c_conv3_3 = self.cl_conv3_3(conv3_3_feats_norm)  # (N, 3 + n_classes - 1, 160, 160)
        # apply Max-out BG label
        max_c, _ = torch.max(c_conv3_3[:, 0:3, :, :], dim=1, keepdim=True)
        c_conv3_3 = torch.cat((max_c, c_conv3_3[:, 3:, :, :]), dim=1)  # (N, n_classes, 160, 160)

        c_conv3_3 = c_conv3_3.permute(0, 2, 3, 1).contiguous()  # (N, 160, 160, n_classes)
        c_conv3_3 = c_conv3_3.view(batch_size, -1, self.n_classes)   # (N, 25600, n_classes)


        c_conv4_3 = self.cl_conv4_3(conv4_3_feats_norm)  # (N, n_classes, 80, 80)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 80, 80, n_classes)
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)   # (N, 6400, n_classes)
    
        c_conv5_3 = self.cl_conv5_3(conv5_3_feats_norm)  # (N, n_classes, 40, 40)
        c_conv5_3 = c_conv5_3.permute(0, 2, 3, 1).contiguous()  # (N, 40, 40, n_classes)
        c_conv5_3 = c_conv5_3.view(batch_size, -1, self.n_classes)   # (N, 1600, n_classes)

        c_conv_fc7 = self.cl_conv_fc7(conv_fc7_feats)  # (N, n_classes, 20, 20)
        c_conv_fc7 = c_conv_fc7.permute(0, 2, 3, 1).contiguous()  # (N, 20, 20, n_classes)
        c_conv_fc7 = c_conv_fc7.view(batch_size, -1, self.n_classes)   # (N, 400, n_classes)

        c_conv6_2 = self.cl_conv6_2(conv6_2_feats)  # (N, n_classes, 10, 10)
        c_conv6_2 = c_conv6_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, n_classes)
        c_conv6_2 = c_conv6_2.view(batch_size, -1, self.n_classes)   # (N, 100, n_classes)

        c_conv7_2 = self.cl_conv7_2(conv7_2_feats)  # (N, n_classes, 5, 5)
        c_conv7_2 = c_conv7_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, n_classes)
        c_conv7_2 = c_conv7_2.view(batch_size, -1, self.n_classes)   # (N, 25, n_classes)

        # Concatenate in this specific order (i.e. must match the order of the prior-boxes!!! )
        locs = torch.cat([l_conv3_3, l_conv4_3, l_conv5_3, l_conv_fc7, l_conv6_2, l_conv7_2], dim=1)  # (N, 34125, 4)
        classes_scores = torch.cat([c_conv3_3, c_conv4_3, c_conv5_3, c_conv_fc7, c_conv6_2, c_conv7_2], dim=1)  # (N, 34125, n_classes)

        return locs, classes_scores



class SSD_Face(nn.Module):
    """
    The SSD_Face network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes):
        super(SSD_Face, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions()

        # 与论文一致
        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 640, 640)
        :return: 34125 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network
        conv3_3_feats, conv4_3_feats, conv5_3_feats, conv_fc7_feats = self.base(image)

        # apply L2 norm
        conv3_3_feats = self.L2Norm3_3(conv3_3_feats)
        conv4_3_feats = self.L2Norm4_3(conv4_3_feats)
        conv5_3_feats = self.L2Norm5_3(conv5_3_feats)

        # Run Auxiliary convolutions
        conv6_2_feats, conv7_2_feats = self.aux_convs(conv_fc7_feats)

        # Run Prediction convolutions(predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv3_3_feats, conv4_3_feats, conv5_3_feats, conv_fc7_feats, conv6_2_feats, conv7_2_feats)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 34125 prior (default) boxes for the SSD_Face, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (34125, 4)
        """

        input_size = 640
        feature_maps = [160, 80, 40, 20, 10, 5]
        anchor_sizes = [16, 32, 64, 128, 256, 512]
        steps = [4, 8, 16, 32, 64, 128]
        imh = input_size
        imw = input_size

        prior_boxes = []

        for k in range(len(feature_maps)):
            feath = feature_maps[k]
            featw = feature_maps[k]
            for i, j in product(range(feath), range(featw)):
                f_kw = imw / steps[k]
                f_kh = imh / steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = anchor_sizes[k] / imw
                s_kh = anchor_sizes[k] / imh

                prior_boxes.append([cx, cy, s_kw, s_kh])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(min=0, max=1)    # (34125, 4)

        return prior_boxes

