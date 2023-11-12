import torch.nn as nn
import torchvision.models as models

from libkaraaba import cv_utils



class CustomNet(nn.Module):
    def __init__(self, num_classes, in_features_size=256, adaptive_pool_output=(1, 1), pretrained=True):  # resnet50
    # def __init__(self, num_classes, in_features_size=1024, adaptive_pool_output=(1, 1), pretrained=True): #resnet50

        super(CustomNet, self).__init__()

        self.custom_model = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-3])

        self.adaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=adaptive_pool_output)

        # print(self.custom_model)
        # new_size = 512*2
        #new_size = 512
        new_size = in_features_size * adaptive_pool_output[0] * adaptive_pool_output[1]
        # print("new_size", new_size)
        # self.resnet18[7][1].bn2 = nn.BatchNorm2d(new_size)
        # # self.resnet18[7][1].conv2 = nn.Conv2d(256+256, 256+256+5, kernel_size=(3,3))
        # self.resnet18[7][1].conv2 = nn.Conv2d(256+256, new_size, kernel_size=(3,3), stride = (1, 1), padding = (1, 1), bias = False)
        #, stride = (1, 1), padding = (1, 1), bias = False

        self.fc = nn.Linear(new_size, num_classes)

        # print(self.custom_model)
        #self.resnet18 = models.resnet18()

    def forward(self, x):
        try:
            x = self.custom_model(x)
            x = self.adaptiveAvgPool2d(x)

            x = x.view(x.size(0), -1)
            x = self.fc(x)
        except Exception as E:
            print("Exception",E)
            print(self.custom_model)
            exit()
        return x