import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def prepare_data(config):
    """
    Prepare train, validation and test data loaders
    """
    # Define transforms
    transform_train = transforms.Compose([             
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),        
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # Load full training dataset with training transforms
    trainval = torchvision.datasets.CIFAR10(
        root=config['paths']['data_dir'],
        train=True,
        download=True,
        transform=transform_train
    )
    
    # Calculate lengths for train-val split
    train_size = int(config['data_config']['train_val_split'] * len(trainval))
    val_size = len(trainval) - train_size
    
    # Split into training and validation sets
    trainset, valset = random_split(trainval, [train_size, val_size])
    
    # Create data loaders
    trainloader = DataLoader(
        trainset,
        batch_size=config['data_config']['batch_size'],
        shuffle=True,
        num_workers=config['data_config']['num_workers'],
        pin_memory=config['data_config']['pin_memory']
    )
    
    valloader = DataLoader(
        valset,
        batch_size=config['data_config']['batch_size'],
        shuffle=False,
        num_workers=config['data_config']['num_workers'],
        pin_memory=config['data_config']['pin_memory']
    )
    
    # Load test set with test transforms
    testset = torchvision.datasets.CIFAR10(
        root=config['paths']['data_dir'],
        train=False,
        download=True,
        transform=transform_test
    )
    
    testloader = DataLoader(
        testset,
        batch_size=config['data_config']['batch_size'],
        shuffle=False,
        num_workers=config['data_config']['num_workers'],
        pin_memory=config['data_config']['pin_memory']
    )
    
    return trainloader, valloader, testloader