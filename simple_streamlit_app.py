import os.path

import streamlit as st
from PIL import Image

import numpy

import torch
import torch.nn as nn
import torchvision.models as models

class CustomNet(nn.Module):
    def __init__(self, num_classes, in_features_size=256, adaptive_pool_output=(1, 1), pretrained=True):  # resnet18
        super(CustomNet, self).__init__()

        # model_resnet18 = models.resnet18(pretrained=True)
        # torch.save(model_resnet18.state_dict(), 'resnet18_weights.pth')

        self.custom_model = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-3])

        # if pretrained:
        #     state_dict = torch.load('resnet18_weights.pth')
        #     self.custom_model.load_state_dict(state_dict)

        self.adaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=adaptive_pool_output)

        new_size = in_features_size * adaptive_pool_output[0] * adaptive_pool_output[1]
        self.fc = nn.Linear(new_size, num_classes)
        self.adaptive_pool_output = adaptive_pool_output
        self.num_classes = num_classes

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

def main():
    # Title
    st.title("Hello, Streamlit!")

    # Header
    st.header("Welcome to my first Streamlit app")

    # Text
    st.write("This is a simple Streamlit app example.")

    if 'switch' not in st.session_state:
        st.session_state.switch = 0

    image_placeholder = st.empty()
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
       image_org = Image.open(uploaded_file)
       image_placeholder.image(image_org, caption='Brand:',use_column_width=True)


    st.write(os.path.exists("model_24Nov1940-Adam.txt"))
    st.write(numpy.__version__)


if __name__ == "__main__":
    main()
