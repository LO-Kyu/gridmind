import json
import os

notebook_path = r"c:\Projects\gridmind\scripts\gridmind_grpo_colab.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == "code":
        source_text = "".join(cell['source'])
        if "health = requests.get(f\"{ENV_URL}/health\"" in source_text:
            # Replace the health check with a safer one
            new_source = []
            for line in cell['source']:
                if 'health = requests.get(f"{ENV_URL}/health"' in line:
                    new_source.append('    r = requests.get(f"{ENV_URL}", timeout=10)\n')
                    new_source.append('    health = {"status": r.status_code}\n')
                else:
                    new_source.append(line)
            nb['cells'][i]['source'] = new_source
            print(f"Fixed health check in cell {i}")
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
