
import streamlit as st
import requests
from io import BytesIO

from PIL import Image, ImageOps
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

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

    # clamped_outputs = output.clamp(0, 1)

    raw_output = output.cpu().detach().numpy()
    # min_val = np.min(raw_output)
    # max_val = np.max(raw_output)
    # raw_output_norm = (raw_output - min_val)/max_val

    # probabilities = F.softmax(output, dim=0)
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

def add_combos_in_a_row(text_for_combo1='Select option for Combo Box 1',
                        text_for_combo2='Select option for Combo Box 2',
                        options1=('Option 11', 'Option 2', 'Option 3'),
                        options2=('Option AA', 'Option BB', 'Option CC')):
    # Create two columns
    col1, col2 = st.columns(2)

    # Add a combo box to each column
    with col1:
        option1 = st.selectbox(text_for_combo1, options1)
        st.write(f'You selected: {option1}')

    with col2:
        option2 = st.selectbox(text_for_combo2, options2)
        st.write(f'You selected: {option2}')

    return option1,option2

def pil_grayscale(image_rgb_obj):
    image_gs = ImageOps.grayscale(image_rgb_obj)
    rgbimg = Image.merge("RGB", (image_gs, image_gs, image_gs))
    return rgbimg


def main():
    def predict_and_show():
        if color_mode == "RGB" and flip_or_not == "Org":
            # np_value, label, conf = predict_image(uploaded_file, model, labels=classes, res=(w, h))
            np_value, label, conf = predict_image_object(image_org, model, labels=classes, res=(w, h))
            image_placeholder.image(image_org, caption='Brand:' + label +
                                                       " Confidence:" + str(round(conf, 2)),
                                    use_column_width=True)

        elif color_mode == "Grayscale" and flip_or_not == "Org":
            image_gs = ImageOps.grayscale(image_org)
            # rgbimg = Image.new("RGB", image_gs.size)
            rgbimg = Image.merge("RGB", (image_gs, image_gs, image_gs))
            # rgbimg1 = Image.new("RGBA", image_gs.size)
            np_value, label, conf = predict_image_object(rgbimg, model, labels=classes, res=(w, h))
            image_placeholder.image(rgbimg, caption='Brand:' + label + " Confidence:" + str(round(conf, 2)),
                                    use_column_width=True)


        elif color_mode == "RGB" and flip_or_not == "Flip":
            image_rgb_flip = image_org.transpose(Image.FLIP_LEFT_RIGHT)
            np_value, label, conf = predict_image_object(image_rgb_flip, model, labels=classes, res=(w, h))
            image_placeholder.image(image_rgb_flip, caption='Brand:' + label +
                                                            " Confidence:" + str(round(conf, 2)),
                                    use_column_width=True)


        elif color_mode == "Grayscale" and flip_or_not == "Flip":
            image_gs = pil_grayscale(image_org)
            image_gs_flip = image_gs.transpose(Image.FLIP_LEFT_RIGHT)
            np_value, label, conf = predict_image_object(image_gs_flip, model, labels=classes, res=(w, h))
            image_placeholder.image(image_gs_flip, caption='Brand:' + label +
                                                           " Confidence:" + str(round(conf, 2)), use_column_width=True)

    # Title
    st.title("HI, Visitor!")
    # Header
    st.header("Classifier")
    # Text
    st.write("This is a simple Streamlit app example.")

    # model_params_path = "/home/karaaba/workspace/for_github/torchpy/model_24Nov1940-Adam.txt"
    # model_path = "/home/karaaba/workspace/for_github/torchpy/model_24Nov1940-Adam.pth"

    model_params_path = "model_24Nov1940-Adam.txt"
    model_path = "model_24Nov1940-Adam.pth"


    adp_pool,num_classes,classes,resolution = read_model_properties(model_params_path)

    w,h = resolution
    print("w,h", w, h)
    print("classes", list(classes))

    model = CustomNet(num_classes=num_classes, adaptive_pool_output=adp_pool)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() #necessary to disable any drop out units and further
    torch.no_grad()

    # Initialize session state
    if 'previous_url' not in st.session_state:
        st.session_state.previous_url = ''

    #file upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # URL input
    image_url = st.text_input("Or enter an image URL")
    print("st.session_state.previous_url", st.session_state.previous_url)
    print("image_url",image_url)
    # load_button_result = st.button("Load Image from URL")
    # result = st.button("Load Image from URL")
    # print("button result", result)
    color_mode, flip_or_not = add_combos_in_a_row(text_for_combo1="choose color", options1=("RGB", "Grayscale"),
                                                  text_for_combo2="choose flip", options2=("Org", "Flip"))
    image_org = None
    image_placeholder = st.empty()
    try:
        if not image_url == "":
           response = requests.get(image_url)
           image_org = Image.open(BytesIO(response.content))
           predict_and_show()
        elif uploaded_file is not None:
           image_org = Image.open(uploaded_file)
           predict_and_show()
    except:
        pass



    # Load image using OpenCV
    #image = cv2.imread(sys.argv[1])






if __name__ == '__main__':
    main()
