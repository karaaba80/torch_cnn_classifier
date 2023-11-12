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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    train_dataset = datasets.ImageFolder(root=args.train_dir, transform=trans_cust._transform_train(img_width, img_height))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    test_dataset = datasets.ImageFolder(root=args.test_dir, transform=trans_cust._transform_test(img_width, img_height))
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    val_dataset = datasets.ImageFolder(root=args.val_dir, transform=trans_cust._transform_test(img_width, img_height))
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False,  collate_fn=collate_fn)

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

    evaluator.evaluate_load_from_disk(mytrainer.saved_model_name, directory=args.test_dir, device=device, input_shape=(img_width, img_height),
                                      adaptive_pool_output=adaptive_pool_output)
    evaluator.evaluate_load_from_disk(mytrainer.saved_model_name, directory=args.val_dir, device=device, input_shape=(img_width, img_height),
                                      adaptive_pool_output=adaptive_pool_output)


if __name__ == "__main__":
    main()
