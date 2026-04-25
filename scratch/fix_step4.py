import json

file_path = r"c:\Projects\gridmind\scripts\gridmind_grpo_colab.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with id 5e5826e4
for cell in nb['cells']:
    if cell.get('id') == '5e5826e4':
        if len(cell['source']) > 0 and cell['source'][0] == "from transformers import AutoTokenizer, AutoModelForCausalLM\n":
            cell['source'][0] = "import torch\n"
            cell['source'].insert(1, "from transformers import AutoTokenizer, AutoModelForCausalLM\n")
        elif len(cell['source']) > 0 and "from transformers import AutoTokenizer" in cell['source'][0] and "import torch" not in cell['source'][0]:
            cell['source'].insert(0, "import torch\n")
        break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Updated Cell 5e5826e4 successfully.")
