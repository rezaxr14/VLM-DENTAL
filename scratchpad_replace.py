import os

replacements = {
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-2B-Instruct": "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-8B": "Qwen3-VL-8B",
    "Qwen3-VL-2B": "Qwen3-VL-2B",
    "Qwen3-VL": "Qwen3-VL",
    "Qwen3": "Qwen3"
}

files_to_check = []
for root, dirs, files in os.walk("."):
    # Filter dirs in place to prevent scanning large directories
    dirs[:] = [d for d in dirs if d not in [".git", ".venv", "__pycache__", "dental_agent.egg-info", ".pytest_cache"]]
    
    # We skip 'data' unless it's the root 'data' dir (where standalone_agent.py is)
    if root == ".\\data" or root == "data" or "data/" in root or "data\\" in root:
        if root != ".\\data" and root != "data":
            continue
            
    for f in files:
        if f.endswith((".md", ".py", ".yaml", ".ipynb", ".json", ".txt", ".toml")):
            files_to_check.append(os.path.join(root, f))

updated_count = 0
for filepath in files_to_check:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        for old, new in replacements.items():
            new_content = new_content.replace(old, new)
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {filepath}")
            updated_count += 1
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

print(f"Total files updated: {updated_count}")
