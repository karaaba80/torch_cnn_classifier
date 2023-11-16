# torch_cnn_classifier
cnn library for classification

versions of some libraries I use:

torch version is 1.13.1

torchvision version is 0.14.1

opencv-python version is 4.5.2.54

(to install torch and torchvision and some other libraries, see here : https://pytorch.org/get-started/locally/)

to use this program, one needs the dataset split into train/test/validation parts. Use this script similarly as shown above.

python data_splitter.py class_folder1 class_folder2 ... class_folder_n -out output_folder --train_ratio 0.5 --test_ratio 0.2

to train the model:

python train_model_and_test.py --train-dir output_folder/train/ --val-dir output_folder/validation/ --test-dir output_folder/test/ --res 128x128 --bsize 32 --out-model-path model.pth --lr 0.001 --ep 48




