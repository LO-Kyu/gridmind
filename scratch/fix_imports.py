import json

file_path = r"c:\Projects\gridmind\scripts\gridmind_grpo_colab.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == '4cdf0f35':
        source = cell['source']
        for i, line in enumerate(source):
            if 'import time\n' == line or 'import time' in line:
                # Insert right after
                source.insert(i + 1, "import sys\n")
                break
        cell['source'] = source
        print("Updated Step 1 cell")
        break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("All updates applied.")
