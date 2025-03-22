import os
from datetime import datetime

EXCLUDED_DIRS = {"log", "__pycache__", ".git", "scr/utils"}

def get_folder_structure(base_path):
    tree_lines = []
    for root, dirs, files in os.walk(base_path):
        # Modify dirs in-place to skip excluded folders
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        level = root.replace(base_path, "").count(os.sep)
        indent = "│   " * level + "├── "
        tree_lines.append(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            file_indent = "│   " * (level + 1) + "├── "
            tree_lines.append(f"{file_indent}{f}")
    return "\n".join(tree_lines)

def get_all_python_code(base_path):
    code_lines = []
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_path)
                code_lines.append(f"\n\n# ===== File: {rel_path} =====\n")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code_lines.append(f.read())
                except Exception as e:
                    code_lines.append(f"# Error reading {rel_path}: {e}")
    return "\n".join(code_lines)

def save_snapshot(base_path=".", output_dir="log"):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"project_snapshot.txt")

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# === Project Folder Structure ===\n\n")
        out.write(get_folder_structure(base_path))
        out.write("\n\n# === Python Source Code ===\n")
        out.write(get_all_python_code(base_path))

    print(f"✅ Project snapshot saved to {output_file}")

if __name__ == "__main__":
    save_snapshot(base_path=".")
