import torch.nn as nn
from retinanet.utils import conv1x1, conv3x3

class PyramidFeatureNetwork(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatureNetwork, self).__init__()

        ###################################################################
        # TODO: Please substitute the "?" with specific numbers
        ##################################################################
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = conv1x1(C5_size, feature_size)
        self.P5_upsampled = nn.Upsample(scale_factor="?", mode='nearest')
        self.P5_2 = conv3x3(feature_size, feature_size, stride="?")

        # add P5 elementwise to C4
        self.P4_1 = conv1x1(C4_size, feature_size)
        self.P4_upsampled = nn.Upsample(scale_factor="?", mode='nearest')
        self.P4_2 = conv3x3(feature_size, feature_size, stride="?")

        # add P4 elementwise to C3
        self.P3_1 = conv1x1(C3_size, feature_size)
        self.P3_2 = conv3x3(feature_size, feature_size, stride="?")

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = conv3x3(C5_size, feature_size, stride="?")

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = conv3x3(feature_size, feature_size, stride="?")

        ##################################################################

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]