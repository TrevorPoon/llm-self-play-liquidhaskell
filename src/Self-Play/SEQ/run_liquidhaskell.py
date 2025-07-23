import subprocess
import os
import tempfile
import shutil

# Define the Haskell code with Liquid Haskell annotations
haskell_code = """
{-@ LIQUID "--reflection"        @-}
{-@ LIQUID "--ple"               @-}

module MyTest where

import Language.Haskell.Liquid.ProofCombinators

-- Alice’s program P
{-@ reflect double @-}
double :: Int -> Int
double x = x + x

-- Alice proposes Q
{-@ reflect double' @-}
double' :: Int -> Int
double' x = 2 * x

-- Alice must give a proof that ∀x. double x == double' x
{-@ lemma_double_equiv :: x:Int -> { double x == double' x } @-}
lemma_double_equiv :: Int -> Proof
lemma_double_equiv x
  =   double x
  === double' x
  *** QED

"""

haskell_code = """
{-@ LIQUID "--reflection" @-}
{-@ LIQUID "--ple" @-}
module Equiv where

import Language.Haskell.Liquid.ProofCombinators

{-@ reflect addNumbers @-}
addNumbers :: Int -> Int -> Int
addNumbers a b = a + b

{-@ reflect addNumbers' @-}
addNumbers' :: Int -> (Int -> Int)
addNumbers' a = \b -> a + b

-- Alice’s detailed proof of equivalence
lemma_addNumbers_equiv :: Int -> Int -> Proof
lemma_addNumbers_equiv x y
    =   addNumbers x y 
    === addNumbers' x y 
    *** QED
"""

# Define the Haskell file name
haskell_file_name = "MyTest.hs"
# Create a temporary directory
temp_dir = tempfile.mkdtemp()
haskell_file_path = os.path.join(temp_dir, haskell_file_name)

# Write the Haskell code to the file
with open(haskell_file_path, "w") as f:
    f.write(haskell_code)

print(f"Generated {haskell_file_name} in {temp_dir} with Liquid Haskell code.")

# Define the command to run Liquid Haskell directly
# We need to explicitly set the PATH for Z3
liquid_command = [f"ghc", "-fplugin=LiquidHaskell", "-package liquid-prelude", haskell_file_name]

print(f"Running: {' '.join(liquid_command)}")

# Run the Liquid Haskell command directly
try:
    # Ensure the PATH includes where z3 is located for the subprocess
    env = os.environ.copy()
    env['PATH'] = f"{os.path.expanduser('~')}/.local/bin:{env['PATH']}"

    process = subprocess.run(
        liquid_command,
        check=True, capture_output=True, text=True,
        cwd=temp_dir, # Run in the temporary directory
        env=env
    )
    print("\nLiquid Haskell Check Output:")
    print(process.stdout)
    print("Liquid Haskell check completed successfully!")

    # Extract and print Liquid Haskell summary
    for line in process.stdout.splitlines():
        if "LIQUID:" in line:
            print(f"\nLiquid Haskell Summary: {line}")

except subprocess.CalledProcessError as e:
    print(f"\nLiquid Haskell Check Failed with error code {e.returncode}:")
    print(e.stdout)
    print(e.stderr)
    print("\nLiquid Haskell check failed. Please review the errors above.")
except FileNotFoundError:
    print("Error: The executable was not found. Please ensure Liquid Haskell is installed and in your PATH.")
    print(f"Attempted to run: {' '.join(liquid_command)}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Clean up the generated files and directory
    os.remove(haskell_file_path)
    print(f"Cleaned up {haskell_file_name}.")

    # Clean up .hi and .o files
    hi_file = os.path.join(temp_dir, "MyTest.hi")
    o_file = os.path.join(temp_dir, "MyTest.o")
    if os.path.exists(hi_file):
        os.remove(hi_file)
        print(f"Cleaned up {os.path.basename(hi_file)}.")
    if os.path.exists(o_file):
        os.remove(o_file)
        print(f"Cleaned up {os.path.basename(o_file)}.")

    # Clean up the .liquid folder
    liquid_folder = os.path.join(temp_dir, ".liquid")
    if os.path.exists(liquid_folder):
        shutil.rmtree(liquid_folder)
        print(f"Cleaned up {os.path.basename(liquid_folder)} folder.")

    # Remove the temporary directory itself
    shutil.rmtree(temp_dir)
    print(f"Removed temporary directory: {temp_dir}")