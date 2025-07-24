import os

def clean_haskell_files():
    input_dir = "humaneval-hs"
    for filename in os.listdir(input_dir):
        if filename.endswith(".hs"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r') as f:
                content = f.read()
            cleaned_content = content.replace("⭐", "")
            with open(filepath, 'w') as f:
                f.write(cleaned_content)

if __name__ == "__main__":
    clean_haskell_files()
