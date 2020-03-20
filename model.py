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

        return locs, classes_scores   # (N, 34125, 4), (N, 34125, n_classes)

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


    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 34125 locations and class scores (output of ths SSD_Face) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 34125 prior boxes, a tensor of dimensions (N, 34125, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 34125, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 34125, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (34125, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (34125)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (34125)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 34125
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=[0.1, 0.35, 0.5], neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self._th1 = threshold[0]
        self._th2 = threshold[1]
        self._th3 = threshold[2]
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha   

        self.smooth_l1 = nn.L1Loss()   # nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 34125 prior boxes, a tensor of dimensions (N, 34125, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 34125, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device) # (N, 34125, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device) # (N, 34125)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)    # 每张照片中含有多少个目标（人脸）

            overlap = find_jaccard_overlap(boxes[i], self.priors_xy) # (n_objects, 34125)
            
            # ---Stage 1: 经典的锚框匹配方法
            # For each prior, find the object that has the maxximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0) # (34125)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this-
            # First, find the prior that has the maximum overlap for each object.
            overlap_for_each_object, prior_for_each_object = overlap.max(dim=1)   # (n_objects)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 2.   # SSD里是1

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior] # (34125)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self._th2] = 0  (34125)

            # ---Stage 2：锚框补偿策略
            N = (torch.sum(overlap_for_each_object >= self._th2) + torch.sum(overlap_for_each_object >= self._th3)) // 2
            overlap_for_each_prior_clone = overlap_for_each_prior.clone()
            add_idx = overlap_for_each_prior_clone.gt(self._th1).eq(overlap_for_each_prior_clone.lt(self._th2))
            overlap_for_each_prior_clone[1 - add_idx] = 0   # 把不满足overlap大于_th1,小于th2的overlap值置为0
            stage2_overlap, stage2_idx = overlap_for_each_prior_clone.sort(descending=True)  # 降序排列

            stage2_overlap = stage2_overlap.gt(_th1)  # 若满足overlap大于th1则置为逻辑1

            if N > 0:
                N = torch.sum(stage2_overlap[:N]) if torch.sum(stage2_overlap[:N]) < N else N
                label_for_each_prior[stage2_idx[:N]] += 1   # 背景标记修改为目标标记，0 + 1 = 1，1表示人脸


            # store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)


        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0   # (N, 34125)


        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 34125)
        # So, if predicted_locs has the shape (N, 34125, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 34125)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 34125)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 34125)
        conf_loss_neg[positive_priors] = 0.  # (N, 34125), positive priors are ignored (never in top n_hard_negatives)

        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 34125), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 34125)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 34125)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss, loc_loss    # alpha = 1
        