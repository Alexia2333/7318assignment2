import torch

# use GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")

    BATCH_SIZE = 256 
    PIN_MEMORY = True
else:
    DEVICE = torch.device("cpu")
    BATCH_SIZE = 64   
    PIN_MEMORY = False

CONFIG = {
    # dateset
    'data_config': {
        'batch_size': BATCH_SIZE,
        'num_workers': 0,
        'pin_memory': PIN_MEMORY,  
        'train_val_split': 0.9,
    },
    
    # train
    'train_config': {
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'device': DEVICE,
        'gradient_clip': 1.0,
        'scheduler': {
            'type': 'CosineAnnealingLR',    
            'T_max': 100,                    
            'eta_min': 0,                    
        },
        'early_stopping': {
            'patience': 20,                  
            'min_delta': 0.001              
        }
    },
    
    #models
    'model_config': {
        'num_classes': 10,
        'basic_cnn': {
            'conv1_out': 6,
            'conv2_out': 16,
            'fc1_out': 120,
            'fc2_out': 84
        },
        'resnet18': {
            'block_config': [2, 2, 2, 2],
            'init_channels': 64
        }
    },
    
    #path
    'paths': {
        'data_dir': './data',
        'save_dir': './checkpoints',
        'best_model_path': 'best_model.pt'
    }
}