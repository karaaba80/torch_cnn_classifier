import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import confusion_matrix, f1_score

import transforms_custom as trans_cust

def evaluate_load_from_disk(saved_model, directory, device,
                            input_shape=(64, 64), num_classes=2, adaptive_pool_output=(1,1)
                            ):  # directory should contain all the classes

    from CustomNet import CustomNet

    dataset = datasets.ImageFolder(root=directory, transform=trans_cust._transform_test(input_shape[0], input_shape[1]))
    model = CustomNet(num_classes=num_classes, adaptive_pool_output=adaptive_pool_output)
    model.load_state_dict(torch.load(saved_model))
    model.eval()
    
    filelist = [item[0] for item in dataset.imgs]
    

    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    model.to(device)
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data, img_list in zip(loader, filelist):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            all_labels.append(labels.cpu().detach().numpy()[0])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted.cpu().detach().numpy()[0])
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('img', img_list, (predicted == labels).cpu().detach().numpy())
    print(' Accuracy: {:.2f}%'.format(100 * correct / total))
    print("all labels", all_labels, all_predictions)


    matrix =  confusion_matrix(all_labels, all_predictions)
    print("confusion matrix:")
    [print( np.array((100*(m/sum(m)))).astype(int) ) for m in matrix]
    print("f1 score: ", f1_score(all_labels, all_predictions))



def evaluate(model, test_loader, device, dataset_name, images_list=None):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        if images_list is None:
          for data in test_loader:
            images, labels = data
                # print('images', images)
                # exit()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        else:

          for data,img_list in zip(test_loader,images_list):
             images, labels = data
             images, labels = images.to(device), labels.to(device)
             outputs = model(images)
             _, predicted = torch.max(outputs.data, 1)
             total += labels.size(0)
             correct += (predicted == labels).sum().item()
             print('img', img_list, (predicted == labels).cpu().detach().numpy())

    print(dataset_name+' Accuracy: {:.2f}%'.format(100 * correct / total))
    return correct / total
