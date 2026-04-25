import json

file_path = r"c:\Projects\gridmind\scripts\gridmind_grpo_colab.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'ceac8c9d':
        source = cell['source']
        for i, line in enumerate(source):
            if 'tokenizer=tokenizer,' in line:
                source[i] = line.replace('tokenizer=tokenizer,', 'processing_class=tokenizer,')
                print("Updated tokenizer to processing_class in Step 6")
                break
        cell['source'] = source
        break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Change applied.")
