import json

file_path = r"c:\Projects\gridmind\scripts\gridmind_grpo_colab.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
        
    # Fix 1: Step 1 cell
    if cell.get('id') == '4cdf0f35':
        source = cell['source']
        
        # Clean up existing imports at the top
        new_source = []
        imports = []
        idx = 0
        while idx < len(source):
            line = source[idx]
            if line.startswith('import requests') or line.startswith('import json') or line.startswith('import sys') or line.startswith('import time'):
                idx += 1
            else:
                break
                
        # Insert the correct sequence
        new_source.append("import requests\n")
        new_source.append("import json\n")
        new_source.append("import sys\n")
        new_source.append("import time\n")
        
        # Append the rest of the cell
        new_source.extend(source[idx:])
        
        cell['source'] = new_source
        print("Updated Step 1 cell imports")

    # Fix 2: Step 7 cell
    if cell.get('id') == 'dac005cc':
        source = cell['source']
        if len(source) > 0 and 'import torch' not in source[0]:
            if source[0].startswith('def run_llm_episode'):
                source.insert(0, "import torch\n\n")
            else:
                source.insert(0, "import torch\n")
            print("Updated Step 7 cell imports")

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("All updates applied.")
