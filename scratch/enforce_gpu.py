import json

file_path = r"c:\Projects\gridmind\scripts\gridmind_grpo_colab.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
        
    source = cell['source']
    source_text = "".join(source)
    
    # 1. Cell 1: Check for dependency installation cell
    if "!pip install trl" in source_text and "✔ All dependencies installed" in source_text:
        # Check if already added
        if "RuntimeError(\"❌ No GPU found!" not in source_text:
            cell['source'].extend([
                "import torch\n",
                "if not torch.cuda.is_available():\n",
                "    raise RuntimeError(\"❌ No GPU found! Go to Runtime → Change runtime type → Select T4 GPU\")\n",
                "print(f\"✔ GPU ready: {torch.cuda.get_device_name(0)}\")\n"
            ])
            print("Updated Cell 1")
            
    # 2. Step 4 cell
    if 'device_map="cuda" if torch.cuda.is_available() else "cpu"' in source_text:
        for i, line in enumerate(source):
            if 'device_map="cuda" if torch.cuda.is_available() else "cpu"' in line:
                source[i] = line.replace('device_map="cuda" if torch.cuda.is_available() else "cpu"', 'device_map="cuda"')
        print("Updated Step 4 cell")

    # 3. Step 7 cell
    if 'inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400).to(model.device)' in source_text:
        for i, line in enumerate(source):
            if 'inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400).to(model.device)' in line:
                source[i] = line.replace('.to(model.device)', '.to("cuda")')
        print("Updated Step 7 cell")

    # 4. Step 6 cell (GRPO config)
    if 'config = GRPOConfig(' in source_text:
        fp16_found = False
        for i, line in enumerate(source):
            if 'fp16=True,' in line:
                fp16_found = True
                break
        
        if not fp16_found:
            # Add fp16=True, after max_steps=60, or just inside config
            for i, line in enumerate(source):
                if 'config = GRPOConfig(' in line:
                    source.insert(i + 1, "    fp16=True,\n")
                    break
            print("Added fp16=True, to Step 6 cell")
        else:
            print("fp16=True, already present in Step 6 cell")

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("All updates applied.")
