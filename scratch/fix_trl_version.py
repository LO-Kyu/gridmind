import json

file_path = r"c:\Projects\gridmind\scripts\gridmind_grpo_colab.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Replace the first code cell's source
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = [
            "!pip install trl transformers accelerate datasets unsloth requests pandas matplotlib\n",
            "import os\n",
            "os.makedirs('results', exist_ok=True)\n",
            "print(\"✔ All dependencies installed\")\n"
        ]
        break  # Only replace the very first code cell

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Updated Cell 1 successfully.")
