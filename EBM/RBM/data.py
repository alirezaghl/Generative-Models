from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_dataloader(batch_size, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)) 
    ])
    
    dataset = datasets.MNIST(
        root='./data', 
        train=train, 
        transform=transform, 
        download=True
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train
    )
    
    return dataloader
