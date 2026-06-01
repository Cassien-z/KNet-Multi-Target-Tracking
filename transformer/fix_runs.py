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

eval_files = glob.glob('scripts/evaluation/*.py')
eval_replacements = [
    ('os.path.join(SCRIPT_DIR, "run_2026', 'os.path.join(WORKSPACE_DIR, "results", "runs", "run_2026'),
    ('os.path.join(script_dir, "run_2026', 'os.path.join(workspace_dir, "results", "runs", "run_2026')
]
for f in eval_files:
    replace_in_file(f, eval_replacements)

print("Replacement complete.")
