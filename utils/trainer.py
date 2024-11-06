# import torch
# from tqdm import tqdm
# import numpy as np

# class Trainer:
#     def __init__(self, model, criterion, optimizer, device, config):
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.device = device
#         self.config = config
        
#     def train_epoch(self, dataloader):
#         self.model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training')
#         for i, (inputs, labels) in pbar:
#             inputs, labels = inputs.to(self.device), labels.to(self.device)
            
#             self.optimizer.zero_grad()
#             outputs = self.model(inputs)
#             loss = self.criterion(outputs, labels)
#             loss.backward()
            
#             # Gradient clipping if configured
#             if self.config['train_config'].get('gradient_clip'):
#                 torch.nn.utils.clip_grad_norm_(
#                     self.model.parameters(), 
#                     self.config['train_config']['gradient_clip']
#                 )
            
#             self.optimizer.step()
            
#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
            
#             # Update progress bar
#             pbar.set_postfix({
#                 'loss': running_loss/(i+1),
#                 'acc': 100.*correct/total
#             })
        
#         return running_loss/len(dataloader), correct/total

#     def evaluate(self, dataloader):
#         self.model.eval()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for inputs, labels in dataloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
                
#                 running_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
        
#         return running_loss/len(dataloader), correct/total

#     def train(self, trainloader, valloader):
#         epochs = self.config['train_config']['epochs']
#         best_val_acc = 0.0
        
#         for epoch in range(epochs):
#             print(f"\nEpoch {epoch+1}/{epochs}")
            
#             # Train phase
#             train_loss, train_acc = self.train_epoch(trainloader)
            
#             # Validation phase
#             val_loss, val_acc = self.evaluate(valloader)
            
#             print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
#             print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
#             # Save best model
#             if val_acc > best_val_acc:
#                 best_val_acc = val_acc
#                 torch.save(
#                     self.model.state_dict(), 
#                     self.config['paths']['best_model_path']
#                 )
#                 print("Saved new best model")
        
#         return best_val_acc



import torch
import torch.nn as nn
from tqdm import tqdm
import time

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        self.data_stream = torch.cuda.Stream()

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        self.start_event.record()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            with torch.cuda.stream(self.data_stream):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            
            torch.cuda.current_stream().wait_stream(self.data_stream)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            if self.config.get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': total_loss / (batch_idx + 1),
                'Acc': 100. * correct / total
            })
        
        self.end_event.record()
        torch.cuda.synchronize()
        
        epoch_time = self.start_event.elapsed_time(self.end_event) / 1000
        
        return total_loss / len(train_loader), 100. * correct / total, epoch_time

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(val_loader, desc='Validating'):
            with torch.cuda.stream(self.data_stream):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            
            torch.cuda.current_stream().wait_stream(self.data_stream)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total


    def train(self, train_loader, val_loader):
        best_acc = 0
        train_times = []
        num_epochs = self.config['train_config']['epochs']

        for epoch in range(num_epochs):

            print(f'\nEpoch: {epoch+1}/{num_epochs}')

            train_loss, train_acc, epoch_time = self.train_epoch(train_loader)
            train_times.append(epoch_time)
            
            val_loss, val_acc = self.evaluate(val_loader)
            
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f'Learning Rate: {current_lr:.6f}')
            
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
            

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_acc': best_acc,
                }, 'best_model.pth'
                )
                print("Saved new best model")
            
            
            
        return best_acc