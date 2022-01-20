import torchvision
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import torchvision.models as models


classes = ["0.christmas_cookies", "1.christmas_presents", "2.christmas_tree", "3.fireworks", "4.penguin", "5.reindeer",
           "6.santa", "7.snowman"]


checkpoint = torch.load('best_model_TL1.pth.tar')
#print(checkpoint)
resnet18_model = models.resnet152()
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 8
resnet18_model.fc = nn.Linear(num_ftrs,number_of_classes)
resnet18_model.load_state_dict(checkpoint['model'])
torch.save(resnet18_model, 'best_model.pth')

model_sample = torch.load('best_model.pth')

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((-0.1297,-0.2573,-0.3357), (0.4444,0.4226,0.4164))
])


def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    print(classes[predicted.item()])




classify(model_sample, image_transforms, "./data/val/63.png", classes)

