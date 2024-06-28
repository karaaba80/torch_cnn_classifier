import os.path

import streamlit as st
from PIL import Image, ImageOps

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

def add_combos_in_a_row(text_for_combo1='Select option for Combo Box 1',
                        text_for_combo2='Select option for Combo Box 2',
                        text_for_combo3='Select option for Combo Box 2',
                        options1=('Option 11', 'Option 2', 'Option 3'),
                        options2=('Option AA', 'Option BB', 'Option CC'),
                        options3=('Option AA', 'Option BB', 'Option CC')):
    # Create two columns
    col1, col2, col3 = st.columns(3)

    # Add a combo box to each column
    with col1:
        option1 = st.selectbox(text_for_combo1, options1)
        st.write(f'You selected: {option1}')

    with col2:
        option2 = st.selectbox(text_for_combo2, options2)
        st.write(f'You selected: {option2}')

    with col3:
        option3 = st.selectbox(text_for_combo3, options3)
        st.write(f'You selected: {option3}')

    return option1,option2,option3


def add_sliders_in_a_row(text_for_slide1='Select option for Combo Box 1',
                         text_for_slide2='Select option for Combo Box 2'):

    # Create two columns
    col1, col2 = st.columns(2)

    # Add a combo box to each column
    with col1:
        slide1 = st.slider(text_for_slide1, 1, 30, 1)

    with col2:
        slide2 = st.slider(text_for_slide2, 1, 30, 1)

    return slide1, slide2


def read_model_properties(model_params_path):
    model_params = open(model_params_path).readlines()
    properties = {}
    for parameter in model_params:
        if parameter.startswith("classes"):
            properties["classes"] = parameter.split(':')[1].strip().split(',')
        elif parameter.startswith("number of classes"):
            properties["number of classes"] = int(parameter.split(':')[1].strip())
        elif parameter.startswith("adaptive pool output"):
            properties["adaptive pool output"] = tuple(map(int, parameter.split(':')[1].strip().split(",")))
        elif parameter.startswith("resolution"):
            properties["resolution"] = parameter.split(':')[1].strip()

    print(properties)

    adp_pool = properties["adaptive pool output"]
    num_classes = properties["number of classes"]
    classes = properties["classes"]
    resolution = list(map(int, properties["resolution"].split("x")))

    return adp_pool,num_classes,classes,resolution


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


def predict_image_object(img_object, model, labels=("acura", "alpha romeo"), res=(128,128), min_prob_threshold=0.75):
    # print("filepath", filepath, "\n")

    import torchvision.transforms.functional as TF
    import torch.nn.functional as F

    import torchvision
    from PIL import Image
    from scipy.special import softmax

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalizer = torchvision.transforms.Normalize(mean=0.5, std=0.5)

    image = img_object
    print("image", image.size)
    model.to(device=device)
    image = image.resize(res)
    # image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
    data = TF.to_tensor(image)
    data = normalizer(data)
    data.unsqueeze_(0)
    data = data.to(device)
    output = model(data)
    print("output", output.cpu().detach().numpy()[0])
    _, predicted = torch.max(output.data, 1)
    predicted_numpy = predicted.cpu().detach().numpy()

    raw_output = output.cpu().detach().numpy()

    probabilities = np.round(softmax(raw_output), 2)
    confidence_value = np.max(probabilities)

    final_predicted_value = labels[predicted]
    if min_prob_threshold > confidence_value:
        final_predicted_value = "unsure"
        pass
    else:
        # print(os.path.basename(filepath), end=" ")
        print("prob", probabilities, "confidence:", confidence_value, "pred:", predicted_numpy, final_predicted_value)
    return predicted_numpy, labels[predicted], confidence_value  # this part is used for the single main


def pil_grayscale(image_rgb_obj):
    image_gs = ImageOps.grayscale(image_rgb_obj)
    rgbimg = Image.merge("RGB", (image_gs, image_gs, image_gs))
    return rgbimg

def main():
    # Title
    st.title("Hello, Streamlit!")

    # Header
    st.header("Welcome to my Car Classifier App")

    # Text
    # st.write("This is a  example.")

    if 'script_run_once' not in st.session_state:
        st.session_state.script_run_once = False

    st.write("session state:" + str(st.session_state))
    # def on_option_select():
    print("st.session_state.script_run_once",st.session_state.script_run_once)

    image_placeholder = st.empty()
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image_org = None
    # print("switch:", st.session_state.switch)
    if uploaded_file is not None:
       image_org = Image.open(uploaded_file)
       if st.session_state.script_run_once is False:
          image_placeholder.image(image_org, caption='Brand:',use_column_width=True)

    #
    # st.write(os.path.exists("model_24Nov1940-Adam.txt"))
    # st.write(np.__version__)

    model_params_path = "model_24Nov1940-Adam.txt"
    model_path = "model_24Nov1940-Adam.pth"

    adp_pool, num_classes, classes, resolution = read_model_properties(model_params_path)
    model = CustomNet(num_classes=num_classes, adaptive_pool_output=adp_pool)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # necessary to disable any drop out units and further
    torch.no_grad()

    w, h = resolution

    
    color_mode, flip_or_not, crop_options = add_combos_in_a_row(text_for_combo1="choose color", options1=("RGB", "Grayscale"),
                                                  text_for_combo2="choose flip", options2=("Org", "Flip"),
                                                  text_for_combo3="choose crop", options3=("Org", "Crop"))
                                                  #text_for_combo3="choose crop", options3=("Org", "Crop10","Crop15","Crop20")


    if color_mode == "Grayscale":
       image_org = pil_grayscale(image_org)
       image_placeholder.image(image_org, caption='Brand:', use_column_width=True)

    if flip_or_not == "Flip":
       image_org = image_org.transpose(Image.FLIP_LEFT_RIGHT)
       image_placeholder.image(image_org, caption='Brand:', use_column_width=True)

    if crop_options == "Crop":
       st.session_state.script_run_once = True
       # percentage_x = st.slider("Crop X Ratio", 1, 1, 30)
       # percentage_y = st.slider("Crop Y Ratio", 1, 1, 30)
       W, H = image_org.size

       percentage_x,percentage_y = add_sliders_in_a_row(text_for_slide1="CropX", text_for_slide2="CropY")

       crop_perc_x = percentage_x/100
       crop_perc_y = percentage_y/100
        # Step 3: Define the percentage crop
       # if percentage == :
       #    crop_perc = 0.1  # 10% crop
       # elif crop_options.endswith("15"):
       #    crop_perc = 0.15  # 10% crop
       # elif crop_options.endswith("20"):
       #     crop_perc = 0.2  # 10% crop
       # print("crop percentage", crop_perc)
       print("crop_perc_x",crop_perc_x,"crop_perc_y",crop_perc_y)
       left = W * crop_perc_x
       upper = H * crop_perc_y
       right = W * (1 - crop_perc_x)
       lower = H * (1 - crop_perc_y)
       crop_box = (left, upper, right, lower)
       image_org = image_org.crop(crop_box)
       image_placeholder.image(image_org, caption='Brand:', use_column_width=True)

    # st.write(color_mode)
    # st.write(color_mode.lower() is "Grayscale".lower())

    if image_org is not None:
       predicted_numpy, label, confidence_value = predict_image_object(image_org, model, labels=("acura", "alpha romeo"), res=(w,h), min_prob_threshold=0.75)
       st.write("Brand", label, "Confidence", confidence_value)

    st.write("classes"+str(classes))

if __name__ == "__main__":
    main()
