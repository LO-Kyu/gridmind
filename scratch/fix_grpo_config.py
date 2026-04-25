import json

file_path = r"c:\Projects\gridmind\scripts\gridmind_grpo_colab.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
        
    source = cell['source']
    source_text = "".join(source)
    
    # Target Step 6 cell
    if 'config = GRPOConfig(' in source_text and 'trainer = GRPOTrainer(' in source_text:
        new_source = []
        for line in source:
            if 'max_new_tokens=100,' in line and 'generation_kwargs' not in line:
                # Skip this line to remove it from GRPOConfig
                continue
            
            if line.strip() == ')' and len(new_source) > 0 and 'reward_funcs=gridmind_reward_fn,' in new_source[-1]:
                # We are at the end of GRPOTrainer block
                new_source.append('    generation_kwargs={"max_new_tokens": 100},\n')
                new_source.append(line)
            else:
                new_source.append(line)
                
        cell['source'] = new_source
        print("Updated Step 6 cell")
        break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("All updates applied.")
