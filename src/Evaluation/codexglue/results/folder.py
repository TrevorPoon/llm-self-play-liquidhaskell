
import json
import shutil
from pathlib import Path

results_dir = Path(__file__).parent

# 1. Get all the adapter path from the json in @/result
# 2. Split them by "/", get the [-2], get a unique list

adapter_folders = set()
json_files_to_move = []

for fpath in results_dir.glob('*.json'):
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
            adapter_path = data.get('setup', {}).get('adapter_path')
            if adapter_path:
                # Get the second to last component for the folder name
                folder_name = adapter_path.strip('/').split('/')[-4] + '_' + adapter_path.strip('/').split('/')[-2]
                if folder_name:
                    adapter_folders.add(folder_name)
                    json_files_to_move.append((fpath, folder_name))
                else:
                    print(f"Skipping {fpath}: adapter_path {adapter_path} does not yield a valid folder name.")
            else:
                print(f"Skipping {fpath}: missing adapter_path")
    except Exception as e:
        print(f"Skipping {fpath}: {e}")

# 3. create folder from the unique list in the @/result
for folder in adapter_folders:
    new_folder_path = results_dir / folder
    new_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Created folder: {new_folder_path}")

# 4. Move the respective json in the folder
for fpath, folder_name in json_files_to_move:
    destination_folder = results_dir / folder_name
    try:
        shutil.move(str(fpath), str(destination_folder / fpath.name))
        print(f"Moved {fpath.name} to {destination_folder}")
    except Exception as e:
        print(f"Could not move {fpath.name} to {destination_folder}: {e}")

# 5. Copy @analyze_by_adapter.py inside each folder as well
analyze_script_path = results_dir / 'analyze_by_adapter.py'

if analyze_script_path.exists():
    for folder in adapter_folders:
        destination_folder = results_dir / folder
        if not (destination_folder / 'analyze_by_adapter.py').exists():
            try:
                shutil.copy(str(analyze_script_path), str(destination_folder / analyze_script_path.name))
                print(f"Copied {analyze_script_path.name} to {destination_folder}")
            except Exception as e:
                print(f"Could not copy {analyze_script_path.name} to {destination_folder}: {e}")
else:
    print(f"Error: {analyze_script_path} not found.")

