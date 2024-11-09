import unittest
import json
import os
from unittest.mock import patch
import matplotlib.pyplot as plt

# Assuming the above class and method are saved in a file named `training_logger.py`
from logger import plot_models_comparison

class TestPlotModelsComparison(unittest.TestCase):
    def setUp(self):
        """Create a sample directory with log files for testing."""
        self.test_dir = 'test_logs'
        os.makedirs(self.test_dir, exist_ok=True)
        self.model_names = ['BasicCNN', 'ResNet18', 'VGG16', 'SimpleUNet']
        # Sample data mimicking actual training logs
        for model_name in self.model_names:
            model_dir = os.path.join(self.test_dir, f"{model_name}_logs")
            os.makedirs(model_dir, exist_ok=True)
            log_data = {
                'train_loss': [0.5 - 0.01*i for i in range(100)],
                'train_acc': [0.5 + 0.01*i for i in range(100)],
                'val_loss': [0.5 - 0.01*i for i in range(100)],
                'val_acc': [0.5 + 0.01*i for i in range(100)]
            }
            with open(os.path.join(model_dir, f'{model_name}_training_log.json'), 'w') as f:
                json.dump(log_data, f)
    
    def test_plot_models_comparison(self):
        """Test the plot_models_comparison function."""
        # Patch plt.show to prevent actual plots during test
        with patch('matplotlib.pyplot.show'):
            plot_models_comparison(self.test_dir)

        # Check if the output file is created
        output_file = os.path.join(self.test_dir, 'models_comparison.png')
        self.assertTrue(os.path.exists(output_file))

    def tearDown(self):
        """Clean up the test directory after tests."""
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

if __name__ == '__main__':
    unittest.main()
