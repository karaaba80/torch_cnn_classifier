import argparse
import os

import cv2
import numpy as np
import torch

from CustomNet import CustomNet

def main_dir():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on custom dataset')
    parser.add_argument('--test-dir', type=str, required=True, help='Path to the testing data directory')
    parser.add_argument('--model-path', type=str, required=True,
                        help='resnet18 resnet50, resnext')
    parser.add_argument("--res", default="128x128",
                         help="Target size for resizing in the format 'wxh' (default: 128x128).")

    args = parser.parse_args()
    mymodel = CustomNet(num_classes=2, adaptive_pool_output=(3,3))
    mymodel.load_state_dict(torch.load(args.model_path))
    mymodel.eval()
    torch.no_grad()

    print(args.test_dir)
    files = [args.test_dir+os.sep+f for f in os.listdir(args.test_dir)]
    weight,height = list(map(int,args.res.split("x")))
    results = [predict_image(f, mymodel, res=(weight,height)) for f in files]
    values = [r[0].cpu().detach().numpy()[0] for r in results]

    print(sum(values)/len(results))

def main_single():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on custom dataset')
    parser.add_argument('--img', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model-path', type=str, required=True, help='path of the model')
    parser.add_argument("--res", default="128x128",
                         help="Target size for resizing in the format 'wxh' (default: 128x128).")

    args = parser.parse_args()

    model = CustomNet(num_classes=2, #to-do this should be gathered from the model config file
                      adaptive_pool_output=(3,3))
    model.load_state_dict(torch.load(args.model_path))
    model.eval() #necessary to disable any drop out units and further
    torch.no_grad()
    weight, height = list(map(int, args.res.split("x")))
    pred,label = predict_image(args.img, model, res=(weight, height))
    print(pred,label)
    img = cv2.imread(args.img, 1)

    # val = np.mean(img[int(img.shape[1]*0.1):int(img.shape[1]*0.6),int(img.shape[0]*0.1):int(img.shape[0]*0.6)]) #for text color
    # color = 255-int(val) #for text color

    # cv2.putText(img, text=label, org=(int(img.shape[1]*0.4),int(img.shape[0]*0.4)),
    #             thickness=1, fontScale=0.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(color,color,color))
    cv2.imshow("class "+label, img)
    cv2.waitKey(0)

def predict_image(filepath, model, labels=("acura", "alpha romeo"), res=(128,128)):
    import torchvision.transforms.functional as TF
    import torchvision
    from PIL import Image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(os.path.basename(filepath),end=" ")
    normalizer = torchvision.transforms.Normalize(mean=0.5, std=0.5)
    image = Image.open(filepath)
    model.to(device=device)
    image = image.resize(res)
    data = TF.to_tensor(image)
    data = normalizer(data)
    data.unsqueeze_(0)
    data = data.to(device)
    output = model(data)
    _, predicted = torch.max(output.data, 1)
    print("raw output", output.cpu().detach().numpy(), "pred:", predicted.cpu().detach().numpy())
    return predicted,labels[predicted]


import sys
if __name__ == '__main__':
    commands = [('single','main_single_image'),
                ('dir','main_1 directory')]

    if len(sys.argv)==1:
       print ('options are',commands)
       exit()

    inputCommand = sys.argv[1]
    commandArgs = sys.argv[1:]

    if sys.argv[1]==commands[0][0]:
        sys.argv = commandArgs
        main_single()
    elif sys.argv[1]==commands[1][0]:
        sys.argv = commandArgs
        main_dir()
    else:
       print ('unknown keyword:',sys.argv[1])
