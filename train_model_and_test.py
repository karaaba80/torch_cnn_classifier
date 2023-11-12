import copy
import os
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, ConcatDataset
import argparse

from torch.utils.data.dataloader import default_collate
from CustomNet import CustomNet
import evaluator
import Trainer
import transforms_custom as trans_cust

from libkaraaba import fileio

def svm_loss(output, target):
    margin = 1  # Margin for the SVM loss
    num_classes = output.size(1)

    # Construct matrix of correct scores
    correct_scores = output[torch.arange(output.size(0)), target].view(-1, 1)

    # Calculate SVM loss
    loss = torch.sum(torch.clamp(output - correct_scores + margin, min=0))
    loss -= margin  # Subtract the margin for the correct class

    # Average the loss
    loss /= output.size(0)

    return loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



from torch.utils.data import Dataset

class CustomFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        from PIL import Image
        # self.file_names = os.listdir(root)
        # self.file_names = fileio.getFilesDictFrom2ndOrderDir(root)
        self.file_names = fileio.getFilePathsFromDirectory(root)
        self.images = [Image.open(f) for f in self.file_names]
        [ print(type(im)) for im in self.images ]
        self.labels = [fileio.getUpperBaseName(path) for path in self.file_names]
        print(self.labels)

        # labels = [l.split(',')[1].strip() for l in lines]
        labels_dict = {}
        labels_set = list(set(self.labels))
        labels_set.sort()

        # print(self.labels)
        # print('labels_set', labels_set, self.images[0].size)
        # exit()
        for i, l in enumerate(labels_set):
            labels_dict[l] = i
        self.labels = [labels_dict[l] for l in self.labels]
        print("self.labels", self.labels)
        # exit()
        # self.class_numbers = 2
        self.class_numbers = len(labels_set)

        self.transform = transform

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        # print("type(image): ", type(image))

        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)


# def get_mean_std(loader):
#     # Compute the mean and standard deviation of all pixels in the dataset
#     num_pixels = 0
#     mean = 0.0
#     std = 0.0
#     for images, _ in loader:
#         batch_size, num_channels, height, width = images.shape
#         # print ("batch size", batch_size)
#         num_pixels += batch_size * height * width
#         mean += images.mean(axis=(0, 2, 3)).sum()
#         std += images.std(axis=(0, 2, 3)).sum()
#
#     mean /= num_pixels
#     std /= num_pixels
#
#     return mean, std


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 with custom net on custom dataset')
    parser.add_argument('--train-dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--val-dir', type=str, required=True, help='Path to the testing data directory')
    parser.add_argument('--test-dir', type=str, required=True, help='Path to the testing data directory')

    parser.add_argument('--out-model-path', dest="model_path", type=str, required=True)

    parser.add_argument("--res", default="64x64",
                        help="Target size for resizing in the format 'wxh' (default: 128x128).")

    parser.add_argument('--ep', dest="number_of_epochs", type=int, default="32")
    parser.add_argument('--lr', dest="learning_rate", type=float, default="0.004")
    parser.add_argument("--bsize", default="2", type=int, help="batch size (default:2).")


    args = parser.parse_args()

    img_width, img_height = map(int, args.res.split('x'))
    batch_size = args.bsize

    collate_fn = lambda x: tuple(x_.to(device) for x_ in default_collate(x))
    #train_dataset = datasets.ImageFolder(root=args.train_dir, transform=trans_cust._transform_train(img_width, img_height))
    #train_dataset = datasets.ImageFolder(root=args.train_dir, transform=trans_cust._transform_train(img_width, img_height))
    #train_labels = [fileio.getUpperBaseName(path[0]) for path in train_dataset.imgs]
    train_dataset = CustomFolderDataset(root=args.train_dir, transform=trans_cust._transform_train(img_width, img_height))
    print ("train dataset", train_dataset.__getitem__(0))

    # indices_class_A = [idx for idx, label in enumerate(train_labels) if label == 'Alpha Romeo']
    # indices_class_B = [idx for idx, label in enumerate(train_labels) if label == 'Acura']

    # Define the batch size and the number of samples to take from each class
    # batch_size = 6
    # samples_per_class = args.bsize // 2  # In this example, 3 samples from each class

    # Create a BatchSampler to balance the batches
    # sampler1 = BatchSampler(
    #     SequentialSampler(indices_class_A), samples_per_class, drop_last=True
    # )
    # sampler2 = BatchSampler(
    #     SequentialSampler(indices_class_B), samples_per_class, drop_last=True
    # )
    # sampler = ConcatDataset([sampler1, sampler2])
    # sampler = sampler1+sampler2

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # train_loader = DataLoader(dataset=train_dataset, batch_sampler=sampler)


    # test_dataset = datasets.ImageFolder(root=args.test_dir, transform=trans_cust._transform_test(img_width, img_height))
    test_dataset = CustomFolderDataset(root=args.test_dir,
                                        transform=trans_cust._transform_test(img_width, img_height))
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    val_dataset = datasets.ImageFolder(root=args.val_dir, transform=trans_cust._transform_test(img_width, img_height))
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False,  collate_fn=collate_fn)
    # print("val dataset", val_dataset.__getitem__(0))
    # exit()
    # print(train_dataset.__getitem__(0)[10].shape)
    # print(val_dataset.__getitem__(0)[10].shape)
    train_class_numbers = len(os.listdir(args.train_dir))
    test_class_numbers = len(os.listdir(args.test_dir))
    val_class_numbers = len(os.listdir(args.val_dir))

    num_classes = None
    if train_class_numbers == test_class_numbers == val_class_numbers:
       num_classes = train_class_numbers
    else:
       print("class numbers are not equal")
       exit()

    adaptive_pool_output = (3,3)
    mynn = CustomNet(num_classes=num_classes, adaptive_pool_output=adaptive_pool_output, pretrained=True)

    mytrainer = Trainer.trainer(mynn, batch_size=batch_size, lrate=args.learning_rate, device=device)
    mytrainer.train(out_model_path=args.model_path, train_loader=train_loader,
                    validation_loader=val_loader, test_loader=test_loader,
                    num_epochs=args.number_of_epochs)

    evaluator.evaluate_without_loader(mytrainer.saved_model_name, directory=args.test_dir, device=device, input_shape=(img_width, img_height),
                                       adaptive_pool_output=adaptive_pool_output)
    evaluator.evaluate_without_loader(mytrainer.saved_model_name, directory=args.val_dir, device=device, input_shape=(img_width, img_height),
                                       adaptive_pool_output=adaptive_pool_output)






if __name__ == "__main__":
    main()