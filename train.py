import torch
import torch.nn as nn
import torch.optim as optim
from models import BasicCNN, ResNet18, VGG16, SimpleUNet
from utils.data_loader import prepare_data
from utils.trainer import Trainer
from config import CONFIG

def train_model_with_name(model_name, config, trainloader, valloader, testloader):
    print(f"\n{'='*20} Training {model_name} {'='*20}")
    
    # build models
    if model_name == "BasicCNN":
        model = BasicCNN(config)
    elif model_name == "ResNet18":
        model = ResNet18(config)
    elif model_name == "VGG16":
        model = VGG16(config)
    elif model_name == "SimpleUNet":
        model = SimpleUNet(config)
    
    device = config['train_config']['device']
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train_config']['learning_rate'],
        weight_decay=config['train_config']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['train_config']['epochs']
    )

    trainer = Trainer(model, criterion, optimizer,scheduler ,device, config)
    
    # Train
    print(f"\nStarting {model_name} training...")
    trainer.train(trainloader, valloader)
    
    # Evaluate
    print(f"\nEvaluating {model_name} on test set...")
    test_loss, test_acc = trainer.evaluate(testloader)
    print(f"{model_name} Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    return test_loss, test_acc

def main():
    # set
    device = CONFIG['train_config']['device']
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # data
    print("\nPreparing data...")
    trainloader, valloader, testloader = prepare_data(CONFIG)
    
    # train
    results = {}
    models = ["BasicCNN", "ResNet18", "VGG16", "SimpleUNet"]
    
    for model_name in models:
        test_loss, test_acc = train_model_with_name(
            model_name, CONFIG, trainloader, valloader, testloader
        )
        results[model_name] = {"loss": test_loss, "acc": test_acc}
    
    # print result 
    print("\n" + "="*50)
    print("Final Results Comparison:")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"{model_name:10s}: Test Acc = {metrics['acc']*100:.2f}%, Test Loss = {metrics['loss']:.4f}")

if __name__ == '__main__':
    main()