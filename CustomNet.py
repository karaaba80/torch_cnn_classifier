import torch.nn as nn
import torchvision.models as models

class CustomNet(nn.Module):
    def __init__(self, num_classes, in_features_size=256, adaptive_pool_output=(1, 1), pretrained=True):  # resnet18
    # def __init__(self, num_classes, in_features_size=1024, adaptive_pool_output=(1, 1), pretrained=True): #resnet50

        super(CustomNet, self).__init__()

        self.custom_model = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-3])

        self.adaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=adaptive_pool_output)

        new_size = in_features_size * adaptive_pool_output[0] * adaptive_pool_output[1]
        self.fc = nn.Linear(new_size, num_classes)

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
