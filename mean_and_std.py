import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# os.listdir('./data/train')
# training_dataset_path = './data/train'
# training_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# train_dataset = torchvision.datasets.ImageFolder(root=training_dataset_path, transform=training_transforms)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)


def get_mean(loader):
    mean = 0.
    total_images_count = 0
    for images, _ in loader:
        images_count_in_a_batch = images.size(0)
        images = images.view(images_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)

        total_images_count += images_count_in_a_batch

    mean /= total_images_count

    print(mean)
    return mean


def get_std(loader):
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        images_count_in_a_batch = images.size(0)
        images = images.view(images_count_in_a_batch, images.size(1), -1)

        std += images.std(2).sum(0)
        total_images_count += images_count_in_a_batch


    std /= total_images_count
    print(std)
    return std

