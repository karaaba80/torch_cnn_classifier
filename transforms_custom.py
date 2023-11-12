import torchvision.transforms as transforms

def _transform_test(img_width, img_height):
    transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),  # Adjust size as needed
        #
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    return transform

def _transform_train(img_width, img_height):
    transform = transforms.Compose([


        transforms.RandomHorizontalFlip(p=0.2+0.3),

        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

        transforms.RandomRotation(degrees=6),  # Add random rotation up to 30 degrees
        # transforms.Resize()
        # transforms.RandomCrop(0.5),
        transforms.Resize((img_width, img_height)),  # Adjust size as needed

        # transforms.CenterCrop((img_width, img_height)),

        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    return transform