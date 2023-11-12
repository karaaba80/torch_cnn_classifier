import os
import shutil
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description='Split data into train, test, and validation sets')
    parser.add_argument('class_folders', nargs='+', help='Input class folders')
    parser.add_argument('--out', dest='output_dir', required=True, help='Input class folders')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train set split ratio (default: 0.7)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test set split ratio (default: 0.2)')
    args = parser.parse_args()

    # Check if class folders exist
    for folder in args.class_folders:
        if not os.path.exists(folder):
           print(f"Error: Class folder '{folder}' does not exist.")
           return

    # Create output directories
    output_dir = args.output_dir
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    validation_dir = os.path.join(output_dir, 'validation')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Define the split ratios
    train_split = args.train_ratio
    test_split = args.test_ratio
    validation_split = 1.0 - train_split - test_split

    # Iterate through class folders
    for folder in args.class_folders:
        print('folder',folder)
        class_name = os.path.basename(folder)
        train_dest = os.path.join(train_dir, class_name)
        test_dest = os.path.join(test_dir, class_name)
        validation_dest = os.path.join(validation_dir, class_name)

        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(test_dest, exist_ok=True)
        os.makedirs(validation_dest, exist_ok=True)

        # Iterate through images in the class folder
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                source_file = os.path.join(folder, filename)
                rand_num = random.random()
                if rand_num < train_split:
                    shutil.copy(source_file, train_dest)
                elif rand_num < train_split + test_split:
                    shutil.copy(source_file, test_dest)
                else:
                    shutil.copy(source_file, validation_dest)

    print('Data split completed successfully.')

if __name__ == '__main__':
    main()
