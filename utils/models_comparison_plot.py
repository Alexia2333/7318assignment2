import os
import json
import matplotlib.pyplot as plt
import sys

def plot_models_comparison(logs_root='logs'):
    """Compare performance of different models and generate visualization"""
    
    # Check if logs directory exists
    if not os.path.exists(logs_root):
        print(f"Error: Directory '{logs_root}' does not exist")
        return

    # Get most recent 4 experiment directories
    log_dirs = [os.path.join(logs_root, d) for d in sorted(os.listdir(logs_root))[-4:] 
               if os.path.isdir(os.path.join(logs_root, d))]
    
    # Initialize data structures
    models_data = {}
    target_models = ['BasicCNN', 'ResNet18', 'VGG16', 'SimpleUNet']
    colors = {
        'BasicCNN': '#1f77b4',  # blue
        'ResNet18': '#d62728',  # red  
        'VGG16': '#2ca02c',     # green
        'SimpleUNet': '#9467bd'  # purple
    }
    
    # Load data from log files
    for log_dir in log_dirs:
        for model_name in target_models:
            log_file = os.path.join(log_dir, f'{model_name}_training_log.json')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    try:
                        data = json.load(f)
                        if all(key in data for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']):
                            models_data[model_name] = data
                    except json.JSONDecodeError:
                        print(f"Warning: Could not read {log_file}")
    
    if not models_data:
        print(f"No valid model data found in {logs_root}")
        return
        
    # Create comparison plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    line_styles = ['-', '--']  # solid for train, dashed for validation
    
    # Plot loss and accuracy curves
    for model_name, data in models_data.items():
        epochs = range(1, len(data['train_loss']) + 1)
        color = colors[model_name]
        
        # Loss comparison
        ax1.plot(epochs, data['train_loss'], color=color, linestyle=line_styles[0], 
                label=f'{model_name} Train', linewidth=2)
        ax1.plot(epochs, data['val_loss'], color=color, linestyle=line_styles[1], 
                label=f'{model_name} Val', linewidth=2)
        
        # Accuracy comparison
        ax2.plot(epochs, [acc for acc in data['train_acc']], color=color, 
                linestyle=line_styles[0], label=f'{model_name} Train', linewidth=2)
        ax2.plot(epochs, [acc for acc in data['val_acc']], color=color, 
                linestyle=line_styles[1], label=f'{model_name} Val', linewidth=2)
    
    # Calculate and plot convergence speed
    convergence_epochs = []
    model_names = []
    for model_name, data in models_data.items():
        final_acc = data['val_acc'][-1]
        convergence_threshold = 0.9 * final_acc
        convergence_epoch = next((i for i, acc in enumerate(data['val_acc']) 
                                if acc >= convergence_threshold), len(data['val_acc']))
        convergence_epochs.append(convergence_epoch)
        model_names.append(model_name)
    
    bar_colors = [colors[model] for model in model_names]
    ax3.bar(model_names, convergence_epochs, color=bar_colors)
    
    # Configure plot styling
    ax1.set_title('Loss Comparison', fontsize=12, pad=20)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Accuracy Comparison', fontsize=12, pad=20)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Epochs to 90% Max Accuracy', fontsize=12, pad=20)
    ax3.set_ylabel('Number of Epochs')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('models_comparison_plot.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Plot saved as 'models_comparison_plot.png'")

    comparison_data = {
        'models': {}
    }
    
    for model_name, data in models_data.items():
        comparison_data['models'][model_name] = {
            'final_metrics': {
                'train_loss': data['train_loss'][-1],
                'val_loss': data['val_loss'][-1],
                'train_acc': data['train_acc'][-1],
                'val_acc': data['val_acc'][-1],
            },
            'convergence_epoch': next((i for i, acc in enumerate(data['val_acc']) 
                                     if acc >= 0.9 * data['val_acc'][-1]), 
                                    len(data['val_acc'])),
            'best_val_acc': max(data['val_acc']),
            'best_val_acc_epoch': data['val_acc'].index(max(data['val_acc'])) + 1,
            'training_history': {
                'train_loss': data['train_loss'],
                'val_loss': data['val_loss'],
                'train_acc': data['train_acc'],
                'val_acc': data['val_acc']
            }
        }
    
    # 保存比较数据
    comparison_file = 'models_comparison_data.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=4)
    
    print(f"Comparison data saved to '{comparison_file}'")

if __name__ == '__main__':
    # Allow custom logs directory through command line argument
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs'
    plot_models_comparison(logs_dir)