import json
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class TrainingLogger:
    def __init__(self, model_name, config):
        self.model_name = model_name
        # 创建日志目录
        self.log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化记录字典
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': [],
            'model_name': model_name,
            'config': config
        }
        
        # 日志文件路径
        self.log_file = os.path.join(self.log_dir, f'{model_name}_training_log.json')
        
    def log_epoch(self,epoch, train_loss, train_acc, val_loss, val_acc, lr ):
        """记录每个epoch的训练信息"""
        self.history['train_loss'].append(float(train_loss))
        self.history['train_acc'].append(float(train_acc))
        self.history['val_loss'].append(float(val_loss))
        self.history['val_acc'].append(float(val_acc))
        self.history['learning_rates'].append(float(lr))

        self.history['config']['train_config']['device'] = str(self.history['config']['train_config']['device'])

        history_to_save = self.history.copy()
        history_to_save['config'] = self.history['config'].copy()
        history_to_save['config']['train_config'] = self.history['config']['train_config'].copy()
        history_to_save['config']['train_config']['device'] = str(self.history['config']['train_config']['device'])
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def plot_training_history(self, save=True):
        """绘制训练历史图表"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 创建2x2的子图
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title(f'{self.model_name} - Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title(f'{self.model_name} - Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # 学习率曲线
        ax3.plot(epochs, self.history['learning_rates'], 'g-')
        ax3.set_title(f'{self.model_name} - Learning Rate')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.log_dir, f'{self.model_name}_training_plots.png'))
            plt.close()
        else:
            plt.show()

@staticmethod
def plot_models_comparison(logs_root='logs'):
    """比较不同模型的性能"""
    # 读取所有模型的日志
    log_dirs = [os.path.join(logs_root, d) for d in sorted(os.listdir(logs_root))[-4:] 
            if os.path.isdir(os.path.join(logs_root, d))]
    
    # 读取所有实验数据
    models_data = {}
    target_models = ['BasicCNN', 'ResNet18', 'VGG16', 'SimpleUNet']
    
    for log_dir in log_dirs:
        for model_name in target_models:
            log_file = os.path.join(log_dir, f'{model_name}_training_log.json')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    try:
                        data = json.load(f)
                        # 确保数据中包含所需的关键字
                        if all(key in data for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']):
                            models_data[model_name] = data
                    except json.JSONDecodeError:
                        print(f"Warning: Could not read {log_file}")

    if not models_data:
        print(f"No valid model data found in {logs_root}")
        return

    # 创建比较图表
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    colors = {
        'BasicCNN': 'blue',
        'ResNet18': 'red',
        'VGG16': 'green',
        'SimpleUNet': 'purple'
    }
    line_styles = ['-', '--']  # 实线用于训练，虚线用于验证

    for model_name, data in models_data.items():
        epochs = range(1, len(data['train_loss']) + 1)
        color = colors[model_name]
        
        # 损失对比
        ax1.plot(epochs, data['train_loss'], color=color, linestyle=line_styles[0],
                label=f'{model_name} Train', linewidth=2)
        ax1.plot(epochs, data['val_loss'], color=color, linestyle=line_styles[1],
                label=f'{model_name} Val', linewidth=2)
        
        # 准确率对比
        ax2.plot(epochs, [acc for acc in data['train_acc']], color=color, 
                linestyle=line_styles[0], label=f'{model_name} Train', linewidth=2)
        ax2.plot(epochs, [acc for acc in data['val_acc']], color=color, 
                linestyle=line_styles[1], label=f'{model_name} Val', linewidth=2)

    # 收敛速度比较
    convergence_epochs = []
    model_names = []
    for model_name, data in models_data.items():
        final_acc = data['val_acc'][-1]
        convergence_threshold = 0.9 * final_acc
        convergence_epoch = next((i for i, acc in enumerate(data['val_acc']) 
                                if acc >= convergence_threshold), len(data['val_acc']))
        convergence_epochs.append(convergence_epoch)
        model_names.append(model_name)

    # 用相应模型的颜色绘制柱状图
    bar_colors = [colors[model] for model in model_names]
    ax3.bar(model_names, convergence_epochs, color=bar_colors)

    # 设置图表属性
    ax1.set_title('Loss Comparison', fontsize=12)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)

    ax2.set_title('Accuracy Comparison', fontsize=12)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)

    ax3.set_title('Convergence Speed', fontsize=12)
    ax3.set_ylabel('Epochs to 90% max accuracy')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(logs_root, 'models_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()