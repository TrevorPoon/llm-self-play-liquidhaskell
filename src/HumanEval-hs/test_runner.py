import json
import os
import subprocess
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run tests for HumanEval Haskell solutions.')
    parser.add_argument('--number', type=int, help='Run a specific HumanEval problem (e.g., --number 0 for HumanEval-0.hs)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    jsonl_file = os.path.join(script_dir, 'humaneval-hs.jsonl')
    solutions_dir = os.path.join(script_dir, 'humaneval-hs')
    temp_dir = os.path.join(script_dir, 'temp_tests')

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    passed_count = 0
    failed_count = 0
    total_count = 0

    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            task_id = entry['task_id']
            test_code = entry['test']

            match = re.search(r'Haskell/(\d+)', task_id)
            if not match:
                print(f"Could not parse task_id: {task_id}")
                continue
            
            problem_id = match.group(1)
            
            if args.number is not None and int(problem_id) != args.number:
                continue
                
            solution_file = os.path.join(solutions_dir, f'HumanEval-{problem_id}.hs')

            if not os.path.exists(solution_file):
                print(f"Solution file not found for {task_id}: {solution_file}")
                continue

            with open(solution_file, 'r') as sol_f:
                solution_code = sol_f.read()

            temp_test_file = os.path.join(temp_dir, f'Test-{problem_id}.hs')
            executable_file = os.path.join(temp_dir, f'Test-{problem_id}')

            # Some tests have their own module declaration, which we should remove to avoid conflicts.
            test_code = re.sub(r'module\s+.*\s+where', '', test_code)
            
            # The tests for 140+ use HUnit and have a different structure
            if int(problem_id) >= 140:
                 # The solution is a function, and the test file is a main that calls it.
                 # The test file already has `main`
                combined_code = solution_code + "\n" + test_code
            else:
                # For older tests, we need to provide the main function from the solution if it's not in the test
                if "main :: IO ()" not in solution_code:
                     combined_code = solution_code + "\n" + test_code
                else: # solution has a main, test is just checks. Let's see if we can make this work
                    # This case is tricky. The provided jsonl seems to have main for all of them.
                    # Let's assume the test code is self-sufficient with the solution code prepended.
                    combined_code = solution_code + "\n" + test_code

            with open(temp_test_file, 'w') as temp_f:
                temp_f.write(combined_code)

            compile_command = ['/usr/bin/ghc', '-i'+solutions_dir, '-main-is', f'Test-{problem_id}', temp_test_file, '-o', executable_file]
            
            # For HUnit tests, we need to make sure GHC knows where to find the module if it's not standard
            # Also, we need to handle module conflicts. Let's try a simpler approach first.
            # Let's just create one file with the combined code.
            
            if "import Test.HUnit" in test_code:
                 # The provided test code for >= 140 doesn't seem to be a full file, but snippets.
                 # The test code from 140 onwards has its own main and imports.
                 # Let's just create the file with the solution and test code.
                 final_code = solution_code + '\n' + test_code
                 with open(temp_test_file, 'w') as temp_f:
                    temp_f.write(final_code)
                 compile_command = ['/usr/bin/ghc', temp_test_file, '-o', executable_file]
            else:
                 final_code = solution_code + '\n' + test_code
                 with open(temp_test_file, 'w') as temp_f:
                    temp_f.write(final_code)
                 compile_command = ['/usr/bin/ghc', temp_test_file, '-o', executable_file]

            print(f"--- Testing {task_id} ---")
            
            # Compile
            compile_result = subprocess.run(compile_command, capture_output=True, text=True)

            if compile_result.returncode != 0:
                print(f"❌ Compilation failed for {task_id}.")
                print(compile_result.stderr)
                failed_count += 1
                continue
            
            # Run
            run_result = subprocess.run([executable_file], capture_output=True, text=True)

            if run_result.returncode == 0 and not run_result.stderr:
                 # For HUnit, output needs to be checked for failures
                if "import Test.HUnit" in test_code:
                    if "failures" in run_result.stdout.lower() and "0" not in run_result.stdout.lower():
                         print(f"❌ Test failed for {task_id} (HUnit).")
                         print(run_result.stdout)
                         failed_count += 1
                    else:
                         print(f"✅ Test passed for {task_id}.")
                         passed_count += 1
                else:
                    print(f"✅ Test passed for {task_id}.")
                    passed_count += 1
            else:
                print(f"❌ Test failed for {task_id}.")
                if run_result.stdout:
                    print("Stdout:")
                    print(run_result.stdout)
                if run_result.stderr:
                    print("Stderr:")
                    print(run_result.stderr)
                failed_count += 1

    print("\n--- Summary ---")
    print(f"Total tests: {total_count}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"Not found/skipped: {total_count - passed_count - failed_count}")

    # Clean up
    subprocess.run(['rm', '-rf', temp_dir])

if __name__ == '__main__':
    main()
