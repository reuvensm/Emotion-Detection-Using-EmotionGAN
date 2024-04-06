from torchvision import transforms

train_transformations = transforms.Compose([
   
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ColorJitter(brightness=.3),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
