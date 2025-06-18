import subprocess
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_haskell.py <haskell_file>")
        sys.exit(1)

    haskell_file = sys.argv[1]
    print(f"Debug: Python script started for {haskell_file}")

    if not os.path.exists(haskell_file):
        print(f"Error: File not found: {haskell_file}")
        sys.exit(1)

    # Compile Haskell file in-place
    compile_command = [
        "/usr/bin/ghc",
        haskell_file
    ]

    print(f"Debug: Running compile command: {' '.join(compile_command)}")

    try:
        compile_result = subprocess.run(
            compile_command,
            check=True,
            capture_output=True,
            text=True
        )
        print("Debug: Compilation successful.")
        if compile_result.stdout:
            print("Debug: Compiler stdout:\n" + compile_result.stdout)
        if compile_result.stderr:
            print("Debug: Compiler stderr:\n" + compile_result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error during compilation: {e}")
        if e.stdout:
            print("Stdout: " + e.stdout)
        if e.stderr:
            print("Stderr: " + e.stderr)
        sys.exit(1)

    # Determine executable name (default: source file name without .hs)
    exe_name = os.path.join("tmp", os.path.splitext(os.path.basename(haskell_file))[0])
    exe_path = os.path.join(os.getcwd(), exe_name)

    if not os.path.exists(exe_path):
        print(f"Error: Executable '{exe_name}' not found after compilation.")
        sys.exit(1)

    # Run the executable
    print(f"Debug: Running executable: ./{exe_name}")
    try:
        run_result = subprocess.run(
            ["./" + exe_name],
            check=True,
            capture_output=True,
            text=True
        )
        print("Debug: Execution successful.")
        print("--- Haskell Program Output ---")
        print(run_result.stdout.strip())
        print("------------------------------")
        if run_result.stderr:
            print("Execution Stderr:\n" + run_result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        if e.stdout:
            print("Stdout: " + e.stdout)
        if e.stderr:
            print("Stderr: " + e.stderr)
        sys.exit(1)

    print("Debug: Python script finished.")

if __name__ == "__main__":
    main()
