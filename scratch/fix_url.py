import os

directory = r"c:\Projects\gridmind"
old_url = "prajwal782007-gridmind.hf.space"
new_url = "prajwal782007-gridmind.hf.space"

def replace_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return # Skip binary or non-utf-8 files

    if old_url in content:
        new_content = content.replace(old_url, new_url)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

for root, dirs, files in os.walk(directory):
    if '.git' in root or '.venv' in root or 'node_modules' in root:
        continue
    for file in files:
        filepath = os.path.join(root, file)
        replace_in_file(filepath)
