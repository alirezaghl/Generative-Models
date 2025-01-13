import torch
from torchvision import datasets, transforms

batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def binarize_image(tensor):
    return (tensor > 0.5).float()

def get_data_loaders():
    
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(binarize_image)
    ])
    
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=tensor_transform
    )
    
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=tensor_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader, train_dataset, test_dataset

train_loader, test_loader, train_dataset, test_dataset = get_data_loaders()