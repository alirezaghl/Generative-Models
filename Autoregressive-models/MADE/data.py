import torch
from torchvision import datasets, transforms

def binarize(x):
    return (x > 0.5).float()

def get_mnist_loaders(batch_size=16):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)),
        transforms.Lambda(lambda x: binarize(x))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
    
    return train_loader, test_loader, train_dataset, test_dataset