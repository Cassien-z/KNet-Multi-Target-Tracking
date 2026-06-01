import os
import glob

def replace_in_file(path, replacements):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='gbk') as f:
            content = f.read()
            
    original_content = content
    for old, new in replacements:
        content = content.replace(old, new)
        
    if content != original_content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

# Data generation scripts
data_gen_files = glob.glob('scripts/data_generation/*.py')
data_gen_replacements = [
    ('from ecef_utils import', 'from src.utils.ecef_utils import'),
    ('sys.path.insert(0, os.path.dirname(__file__))', 'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))'),
    ('save_dir="train_data_', 'save_dir="data/train_data/train_data_'),
    ('save_dir = "train_data_', 'save_dir = "data/train_data/train_data_'),
    ('save_dir = "../test_data', 'save_dir = "data/test_data'),
    ('os.path.join(SCRIPT_DIR, "train_data_multitarget_1ms")', 'os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "data", "train_data", "train_data_multitarget_1ms")'),
    ('os.path.join(SCRIPT_DIR, "test_data")', 'os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "data", "test_data")'),
    ('\\test_data\\test_data_cssw', '\\data\\test_data\\test_data_cssw')
]
for f in data_gen_files:
    replace_in_file(f, data_gen_replacements)

# Evaluation scripts
eval_files = glob.glob('scripts/evaluation/*.py')
eval_replacements = [
    ('WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)', 'WORKSPACE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))'),
    ('os.path.dirname(os.path.abspath(__file__))', 'os.path.dirname(os.path.abspath(__file__))'),
    ('from dataset import', 'from src.dataset import'),
    ('from model import', 'from src.model import'),
    ('WORKSPACE_DIR, "test_data"', 'WORKSPACE_DIR, "data", "test_data"'),
    ('WORKSPACE_DIR, "train_data"', 'WORKSPACE_DIR, "data", "train_data"'),
    ('workspace_dir, "train_data"', 'workspace_dir, "data", "train_data"'),
    ('\\\\transformer\\\\test_data\\\\', '\\transformer\\data\\test_data\\'),
    ('\\\\transformer\\\\train_data\\\\', '\\transformer\\data\\train_data\\'),
    ('results/eval_compare.py', 'scripts/evaluation/eval_compare.py'),
    ('model_predictions.npy', 'results/model_predictions.npy'),
    ('ground_truth.npy', 'results/ground_truth.npy'),
    ('P_history.npy', 'results/P_history.npy')
]
for f in eval_files:
    replace_in_file(f, eval_replacements)

print("Replacement complete.")
