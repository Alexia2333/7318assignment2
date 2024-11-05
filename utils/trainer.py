import torch
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, criterion, optimizer, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training')
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping if configured
            if self.config['train_config'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['train_config']['gradient_clip']
                )
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss/(i+1),
                'acc': 100.*correct/total
            })
        
        return running_loss/len(dataloader), correct/total

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return running_loss/len(dataloader), correct/total

    def train(self, trainloader, valloader):
        epochs = self.config['train_config']['epochs']
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train phase
            train_loss, train_acc = self.train_epoch(trainloader)
            
            # Validation phase
            val_loss, val_acc = self.evaluate(valloader)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    self.model.state_dict(), 
                    self.config['paths']['best_model_path']
                )
                print("Saved new best model")
        
        return best_val_acc