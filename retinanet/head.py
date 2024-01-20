import torch.nn as nn
import torch
from retinanet.utils import conv1x1, conv3x3

class RegressionHead(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionHead, self).__init__()

        self.conv1 = conv3x3(num_features_in, feature_size)
        self.act1 = nn.ReLU()

        self.conv2 = conv3x3(feature_size, feature_size)
        self.act2 = nn.ReLU()

        self.conv3 = conv3x3(feature_size, feature_size)
        self.act3 = nn.ReLU()

        self.conv4 = conv3x3(feature_size, feature_size)
        self.act4 = nn.ReLU()

        self.output = conv3x3(feature_size, num_anchors * 4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationHead(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = conv3x3(num_features_in, feature_size)
        self.act1 = nn.ReLU()

        self.conv2 = conv3x3(feature_size, feature_size)
        self.act2 = nn.ReLU()

        self.conv3 = conv3x3(feature_size, feature_size)
        self.act3 = nn.ReLU()

        self.conv4 = conv3x3(feature_size, feature_size)
        self.act4 = nn.ReLU()

        self.output = conv3x3(feature_size, num_anchors * num_classes)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)