# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import random
import logging
import argparse
import subprocess
import shutil
from tqdm import tqdm
from peft import PeftModel
import vllm
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import textwrap
import tempfile
from datasets import load_dataset, load_from_disk, Dataset
from typing import List
import re

from utils.utils import get_function_name, print_nvidia_smi, print_gpu_memory_usage, strip_comments, get_function_arg_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Prompts (from SEQ Paper, Appendix C) ---

ALICE_EQUIV_SYSTEM_PROMPT = textwrap.dedent("""
    You are a helpful and expert Haskell programmer, powered by Liquid Haskell.
    Your job is to *transform* any given function `P` into a new function `Q` that:

      • Is syntactically correct Haskell.  
      • Is semantically equivalent: ∀x. `P x == Q x`.  
      • Uses a *different* implementation -- Implement a non-structural change rather than merely swapping operator order. 
      • Uses a different function name (e.g. add a trailing `'_alt`).  

    The original program and your program will be used in a test to evaluate the skill of an expert Haskell programmer who will identify whether your program Q is semnatic equivalent to the original program P, so make sure that the difference you introduce are not very easy to understand. 
    You will be given a difficulty level from 0 (easiest) to 10 (hardest) to target. E.g. difficulty level 0 means that an expert computer scientist in the bottom decile or above should be able to find a diverging example, difficulty level 9 means that only an expert computer scientist in the top decile should be able to find a diverging example, and difficulty level 10 means that only the top 0.01 or less of expert Haskell programmer should be able to find a diverging example.                                 

    Always think through your transformation steps in `<think>…</think>`, then emit exactly:

    **Generated Program `Q`:**
    ```haskell
    <your Q here>
    ```
""").strip()

ALICE_EQUIV_USER_PROMPT = textwrap.dedent("""
    Difficulty level: {difficulty_level}
    Here is the original Haskell function `P`:
    ```haskell
    {program_p_completion}
    ```

    Its argument type is  
    ```haskell
    t = {t}
    ```

    **Your task**: produce a new function `Q` that satisfies the system prompt requirements.  
    - Make sure `Q` has a different name (e.g. append a `'_alt`).  
    - Avoid trivial symmetric rewrites—show a genuine alternative implementation.  
    - Do not include any extra commentary beyond the required `<think>…</think>` and the `**Generated Program `Q`:** block.
    - Where appropriate, feel free to use Prelude functions such as foldr, map, or zipWith to encourage diverse strategies.

    <think>
""").strip()

ALICE_INEQUIV_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Haskell programmer. Your task is to generate a semantically inequivalent variant of a given Haskell program, which means that there must exist at least a diverging input example such that the original program and your program either produce different outputs or exceptions, or one halts and the other one does not halt.
    You must also provide a diverging input, which is a valid input for both programs, but on which they produce different outputs.
                                      
    A good inequivalent program `Q` should be subtly different from `P`.
    A good diverging input `x` should be simple and clearly demonstrate the semantic difference between `P` and `Q`.

    The original program and your program will be used in a test to evaluate the skill of an expert Haskell programmer who will have to produce a diverging example (not necessarily the same as yours), so make sure that the difference you introduce are not very easy to understand. 
    You will be given a difficulty level from 0 (easiest) to 10 (hardest) to target. E.g. difficulty level 0 means that an expert computer scientist in the bottom decile or above should be able to find a diverging example, difficulty level 9 means that only an expert computer scientist in the top decile should be able to find a diverging example, and difficulty level 10 means that only the top 0.01 or less of expert Haskell programmer should be able to find a diverging example.                                 

    First, think step-by-step and write down your analysis of program `P` and your strategy for creating an inequivalent program `Q`. Enclose this reasoning within `<think>` and `</think>` tags.
    After the thinking block, the final answer could **only** be in the following format, without any additional explanation or context.

    Final output MUST be exactly: 
    **Generated Program `Q`:**
    ```haskell
    <Your generated Haskell code for `Q`>
    ```

    **Diverging Input `x`:**
    ```
    <The diverging input `x`>
    ```

""").strip()

ALICE_INEQUIV_USER_PROMPT = textwrap.dedent("""
    Difficulty level: {difficulty_level}
    Original program `P`:
    ```haskell
    {program}
    ```

    <think>
""").strip()

ALICE_DIFFICULTY_PREDICTION_EQUIVALENT_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""
    Difficulty level: Any
    Program P
    ```haskell
    {program_p}
    ```
    The program Q below is semantically equivalent to the original program P, where the equivalence is due to the fact that the two programs have the same behavior on all inputs.
    Program Q
    ```haskell
    {program_q}
    ```
""").strip()

ALICE_DIFFICULTY_PREDICTION_INEQUIVALENT_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""
    Difficulty level: Any
    program P
    ```haskell  
    {program_p}
    ```
    The program Q below is semantically inequivalent to the original program P, where the inequivalence is due to the fact that the two programs have different behavior on some inputs.
    Program Q
    ```haskell
    {program_q}
    ```
""").strip()

ALICE_DIFFICULTY_PREDICTION_USER_PROMPT = textwrap.dedent("""
    Predict the difficulty level of the instance of Program Q compared to Program P. Just write \"Difficulty level: D\" where D is your prediction, do not write anything else.
""").strip()

BOB_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Haskell programmer. You are given two Haskell programs, `P` and `Q`.
    Your task is to determine if they are semantically equivalent.
    Use the following format to respond:
    # Equivalent?
    Yes or No

    If the programs are equivalent, respond with your thought process and a final output with: 
    # Equivalent?
    Yes

    If they are inequivalent, respond with your thought process and a final output with:
    # Equivalent?
    No
""").strip()

BOB_USER_PROMPT_TEMPLATE = textwrap.dedent("""
    Program `P`:
    ```haskell
    {program_p}
    ```

    Program `Q`:
    ```haskell
    {program_q}
    ```
    <think>
""")

LEMMA_PROOF_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Haskell/Liquid Haskell prover.
    You are asked to prove that two reflectedfunctions are equivalent.
                                            
    The most basic proof should be in the following format:
    ```haskell
    {{-@ lemma_{func_name_p}_equiv :: x:{arg_type} -> {{ {func_name_p} x == {func_name_q} x }} @-}}
            lemma_{func_name_p}_equiv :: {arg_type} -> Proof
            lemma_{func_name_p}_equiv x
                =   {func_name_p} x 
                === {func_name_q} x 
                *** QED
    ```
                                            
    However, you should also use more advanced proof techniques if necessary. 
                                            
    Few-Shot Example 1:
    
    ```haskell
    {{-@ LIQUID "--reflection" @-}}
    {{-@ LIQUID "--ple" @-}}

    module MyTest where

    import Language.Haskell.Liquid.ProofCombinators

    -- Alice program P
    {{-@ reflect double @-}}
    double :: Int -> Int
    double x = x + x

    -- Alice proposes Q
    {{-@ reflect double' @-}}
    double' :: Int -> Int
    double' x = 2 * x

    -- Here is the full lemma, from annotation to QED:
    {{-@ lemma_double_equiv :: x:Int -> {{ double x == double' x }} @- }}
    lemma_double_equiv :: Int -> Proof
    lemma_double_equiv x
    =   double x
    === double' x
    *** QED
    ```
                                            
    Few-Shot Example 2:
    ```haskell
    {{-@ LIQUID "--reflection" @-}}
    {{-@ LIQUID "--ple" @-}}
    module Equiv where

    import Language.Haskell.Liquid.ProofCombinators

    {{-@ reflect addNumbers @-}}
    addNumbers :: Int -> Int -> Int
    addNumbers a b = a + b

    {{-@ reflect addNumbers' @-}}
    addNumbers' :: Int -> (Int -> Int)
    addNumbers' a = \b -> a + b

    -- Alice detailed proof of equivalence
    lemma_addNumbers_equiv :: Int -> Int -> Proof
    lemma_addNumbers_equiv x y
        =   addNumbers x y 
        === addNumbers' x y 
        *** QED
                                            
    When you answer, output **only** the complete lemma block in the same style:
    1. Use the `{{-@ lemma_… @-}}` annotation , with the exact naming pattern lemma_<P>_equiv
    2. The Haskell type signature  
    3. The function definition with `===` steps  
    4. End with `*** QED`
    5. Please put your proof between ```haskell and  ```
    No extra text, no additional comments.
    Your answer must match the example format exactly, without trailing whitespace or newlines outside the code block.                                     

    """).strip()

LEMMA_PROOF_USER_PROMPT = textwrap.dedent("""
                                          
    {error_msg_section}
                                          
    {equiv_code}
    
    ------------------------------------------------------------

    **Your task**: Produce the proof of equivalence for the following function:
    `{func_name_p} x == {func_name_q} x` for all `x`.  

    ```haskell
    {{-@ LIQUID "--reflection" @-}}
    {{-@ LIQUID "--ple" @-}}
    module Equiv where
    import Language.Haskell.Liquid.ProofCombinators

    {{-@ reflect {func_name_p} @-}}
    {program_p_content}

    {{-@ reflect {func_name_q} @-}}
    {program_q_content}

    -- Your complete proof of equivalence
    /* PROOF BODY HERE */
    ```
    <think>
""").strip()
# --- Code Execution ---

class CodeExecutor:
    """A wrapper for executing Haskell code and comparing results."""
    def __init__(self, seq_instance, timeout: float = 60.0):
        self.seq = seq_instance
        self.timeout = timeout
        self.tmp_dir = tempfile.mkdtemp(prefix="haskell_executor_")

    def _create_haskell_program(self, program_code: str, main_body: str) -> str:
        """Builds a complete Haskell source file from a function definition and a main body."""
        prog = textwrap.dedent(program_code).strip()
        body = textwrap.dedent(main_body).rstrip()

        # No imports needed for the initial programs.
        imports = ""

        imports = """
import Prelude
import Data.List
import Data.Char
import Data.Maybe
import Text.Read (readMaybe)
""".strip()

        return f"{imports}\n\n{prog}\n\n{body}"

    def execute(self, program_code: str, input: any):
        """Executes a Haskell program with a given input and returns the stdout."""
        func_name = get_function_name(program_code)
        arg_type = get_function_arg_type(program_code)

        logger.info(f"Function name: {func_name}")
        logger.info(f"Argument type: {arg_type}")
        logger.info(f"Input: {input}")

        if not func_name or not arg_type:
            logger.warning(f"Could not find function name or argument type in program. Assuming not executable.")
            return "no_function", "Not executable"
        
        main_body = textwrap.dedent(f"""
main :: IO ()
main = do
    line <- getLine
    case readMaybe line :: Maybe {arg_type} of
        Just arg -> print ({func_name} arg)
        Nothing -> error "Invalid input: expected {arg_type}"
        """).strip()

        program = self._create_haskell_program(program_code, main_body)

        logger.info(f"Executing Program: \n\n{program}\n\n")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.hs', delete=True, dir=self.tmp_dir) as f:
            f.write(program)
            f.flush()

            try:
                compile_process = subprocess.run(
                    ['/usr/bin/ghc', '-o', f.name.replace('.hs', ''), f.name], 
                    capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                logger.warning("Compilation timed out.")
                return "compile_timeout", "Compilation timed out"
            except Exception as e:
                logger.warning(f"An unexpected error occurred during compilation: {e}")
                return "compile_error", str(e)

            if compile_process.returncode != 0:
                return "compile_error", compile_process.stderr

            try:
                run_process = subprocess.run(
                    [f.name.replace('.hs', '')],
                    input=input,
                    capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                logger.warning("Execution timed out.")
                return "runtime_timeout", "Execution timed out"
            except Exception as e:
                logger.warning(f"An unexpected error occurred during execution: {e}")
                return "runtime_error", str(e)

            if run_process.returncode != 0:
                return "runtime_error", run_process.stderr

            return "success", run_process.stdout.strip()

    def check_divergence(self, p: str, q: str, x: any) -> bool:
        """Checks if two programs diverge on a given input."""
        status_p, out_p = self.execute(p, x)
        status_q, out_q = self.execute(q, x)

        logger.info(f"Status P: {status_p}")
        logger.info(f"Status Q: {status_q}")
        logger.info(f"Output P: {out_p}")
        logger.info(f"Output Q: {out_q}")

        if status_q == "compile_error":
            return False
        
        # Case 1: Both succeeded but outputs are different
        if status_p == "success" and status_q == "success":
            return out_p != out_q

        # Case 2: One succeeded, the other didn't (implies divergence)
        if (status_p == "success" and status_q != "success") or \
           (status_q == "success" and status_p != "success"):
            return True

        # Case 3: Both failed, but in different ways (implies divergence)  
        if status_p != status_q:
            return True

        # If none of the above, they do not diverge (e.g., both compile_error, or both runtime_timeout)
        return False

    def check_compiles(self, program_code: str) -> bool:
        """Checks if a Haskell program compiles successfully."""
        main_body = 'main :: IO ()\nmain = putStrLn "compiles"'
        if 'main = ' in program_code:
            main_body = 'main :: IO ()'
        program = self._create_haskell_program(program_code, main_body)

        logger.info(f"Compiling Program: \n\n{program}\n\n")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.hs', delete=True, dir=self.tmp_dir) as f:
            f.write(program)
            f.flush()
            try:
                compile_process = subprocess.run(
                    ['/usr/bin/ghc', '-c', f.name],
                    capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                logger.warning("GHC type-check timed out.")
                return False
            except Exception as e:
                logger.warning(f"An unexpected error occurred during GHC type-check: {e}")
                return False

            if compile_process.returncode != 0:
                logger.warning(f"GHC type-check error:\n\n{compile_process.stderr}")
                return False
        return True

    def generate_proof_body(self, program_p: str, program_q: str, func_name_p: str, func_name_q: str, arg_type: str, error_msg: str, equiv_code: str) -> str:
        """
        Ask Alice to fill in the proof. If error_msg is set, include it so she can fix mistakes.
        Returns just the body (the lines after '=').
        """
        program_p_content = textwrap.dedent(program_p)
        program_q_content = textwrap.dedent(program_q)
        # 1. Build a prompt that shows the reflected P and Q,
        #    and that asks for a sequence of ===, ?lemmas, and *** QED.

        if error_msg:
            error_msg_section = f"Liquid Haskell failed with:\n```\n{error_msg}\n```\nPlease fix your proof accordingly."
        else:
            error_msg_section = ""

        proof_user_prompt = LEMMA_PROOF_USER_PROMPT.format(
            func_name_p=func_name_p,
            func_name_q=func_name_q,
            program_p_content=program_p_content,
            program_q_content=program_q_content,
            arg_type=arg_type,
            error_msg_section=error_msg_section, 
            equiv_code=equiv_code
        )

        # logger.info(f"Proof system prompt: \n{LEMMA_PROOF_SYSTEM_PROMPT}")
        # logger.info(f"Proof user prompt: \n{proof_user_prompt}")
        
        # Call your LLM (Alice) with this as a single-turn prompt:
        lemma_proof_system_prompt = LEMMA_PROOF_SYSTEM_PROMPT.format(
            func_name_p=func_name_p,
            func_name_q=func_name_q,
            arg_type=arg_type
        )
        response = self.seq.call_alice_for_proof(lemma_proof_system_prompt, proof_user_prompt)
        # logger.info(f"Alice's raw lemma proof generation response: {response}")

        # Check if Alice returned the full function definition

        # 1) Try to grab the haskell‐labelled code block
        pattern_hs = re.compile(r"```haskell\s*([\s\S]*?)```", flags=re.MULTILINE)
        match = pattern_hs.search(response)

        # 2) If none, fall back to any code block
        if not match:
            pattern_any = re.compile(r"```(?:[^\n]*\n)?([\s\S]*?)```", flags=re.MULTILINE)
            match = pattern_any.search(response)

        if match:
            body = match.group(1).strip()
            logger.info(f"Extracted lemma proof body from full definition:\n{body}")
            return body, response.strip()
        elif "</think>" in response:
            post_think = response.partition("</think>")[2].strip()
            return post_think, response.strip()
        else: 
            logger.info(f"Assuming entire response is the proof body.")
            return response.strip(), response.strip()

    def verify_equivalence(self, program_p: str, program_q: str):
        """Verifies semantic equivalence using Liquid Haskell."""
        logger.info("Verifying equivalence with Liquid Haskell...")

        func_name_p = get_function_name(program_p)
        func_name_q = get_function_name(program_q)
        arg_type_p = get_function_arg_type(program_p)
        arg_type_q = get_function_arg_type(program_q)

        logger.info(f"Function name P: {func_name_p}")
        logger.info(f"Function name Q: {func_name_q}")
        logger.info(f"Argument type P: {arg_type_p}")
        logger.info(f"Argument type Q: {arg_type_q}")

        if not func_name_p or not arg_type_p or not func_name_q or not arg_type_q:
            logger.warning("Could not extract function name or argument type for P or Q. Skipping LH verification.")
            return "lh_error", "Function name or arg type extraction failed."
        
        # Check if Q is different from P
        if program_p.strip()== program_q.strip().replace(func_name_q, func_name_p):
            logger.warning("🟨 Alice generated an identical program Q to P. Skipping example.")
            return "identical", "Alice generated an identical program Q to P. Skipping example."

        # Create temporary files for Original.hs and Variant.hs
        original_file_path = os.path.join(self.tmp_dir, "Original.hs")
        dedented_program_p = textwrap.dedent(program_p).strip()
        with open(original_file_path, "w") as f_p:
            content_p = "module Original where\n" \
                        f"{{-@ reflect {func_name_p} @-}}\n" \
                        f"{dedented_program_p}"
            f_p.write(content_p)
        
        variant_file_path = os.path.join(self.tmp_dir, "Variant.hs")
        dedented_program_q = textwrap.dedent(program_q).strip()
        with open(variant_file_path, "w") as f_q:
            content_q = "module Variant where\n" \
                        f"{{-@ reflect {func_name_q} @-}}\n" \
                        f"{dedented_program_q}"
            f_q.write(content_q)

        # Create the Equiv.hs file
        equiv_code_template = textwrap.dedent("""                       
            {{-@ LIQUID "--reflection" @-}}
            {{-@ LIQUID "--ple" @-}}
            module Equiv where

            import Language.Haskell.Liquid.ProofCombinators

            {{-@ reflect {func_name_p} @-}}
            {program_p_content}

            {{-@ reflect {func_name_q} @-}}
            {program_q_content}

            -- Alice's detailed proof of equivalence
            {proof_body}
        """).strip()

        equiv_code_prompt = textwrap.dedent("""
        Below are two reflected definitions with liquid haskell proof in previous attempt:
        ```haskell
        {equiv_code}
        ```
        """) .strip()


        MAX_PROOF_ATTEMPTS = 3 # Set a limit for proof attempts

        # proof_body_template = textwrap.dedent("""
        #     {{-@ lemma_{func_name_p}_equiv :: x:{arg_type_p} -> {{ {func_name_p} x == {func_name_q} x }} @-}}
        #     lemma_{func_name_p}_equiv :: {arg_type_p} -> Proof
        #     lemma_{func_name_p}_equiv x
        #         =   {func_name_p} x 
        #         === {func_name_q} x 
        #         *** QED
        # """).strip()

        # proof_body = proof_body_template.format(
        #     func_name_p=func_name_p,
        #     func_name_q=func_name_q,
        #     arg_type_p=arg_type_p,
        # )

        # equiv_llm_response = ""

        error_msg = ""
        equiv_code = ""
        equiv_code_prompt_msg = ""

        proof_body, equiv_llm_response = self.generate_proof_body(
            program_p, program_q, func_name_p, func_name_q, arg_type_p, error_msg, equiv_code_prompt_msg
        )

        for attempt in range(MAX_PROOF_ATTEMPTS):
            logger.info(f"--- Proof attempt {attempt + 1}/{MAX_PROOF_ATTEMPTS} ---")
            
            
            if proof_body.strip() == "":
                logger.warning("🟥 --- Alice generated an empty proof body. Skipping example. ---")
                return "empty_proof", "Alice generated an empty proof body. Skipping example."

            # 2. Render Equiv.hs with proof_body
            equiv_code = equiv_code_template.format(
                func_name_p=func_name_p,
                func_name_q=func_name_q,
                program_p_content=dedented_program_p,
                program_q_content=dedented_program_q,
                proof_body=proof_body
            )

            equiv_code_prompt_msg = equiv_code_prompt.format(
                equiv_code=equiv_code
            )

            equiv_file_path = os.path.join(self.tmp_dir, "Equiv.hs")
            with open(equiv_file_path, 'w') as f_equiv:
                f_equiv.write(equiv_code)
            
            try:
                # 3. Run Liquid Haskell
                env = os.environ.copy()
                env['PATH'] = f"{os.path.expanduser('~')}/.local/bin:{env['PATH']}"
                
                lh_process = subprocess.run(
                    [
                        'ghc',
                        '-fplugin=LiquidHaskell',
                        '-package',
                        'liquid-prelude',
                        equiv_file_path
                    ],
                    capture_output=True, text=True, timeout=self.timeout,
                    cwd=self.tmp_dir, 
                    env=env
                )
                
                logger.info(f"Equiv.hs (Attempt {attempt + 1}): \n{equiv_code}")
                
                if lh_process.returncode == 0:
                    logger.info("✅ Liquid Haskell proof accepted.")

                    # Add to Alice's training data (proof-conditioned)
                    liquid_haskell_proof_system_prompt = textwrap.dedent("""
                        You are a helpful assistant that generates Liquid Haskell proofs.
                        Given two program P and Q, you will be asked to generate a proof that P and Q are equivalent.                                                                        
                    """).strip()

                    liquid_haskell_proof_user_prompt = textwrap.dedent("""
                        Here are the two programs:
                        Program P:
                        ```haskell
                        {program_p}
                        ```

                        Program Q:  
                        ```haskell
                        {program_q}
                        ```

                        You will be asked to generate a proof that P and Q are equivalent.
                    """).strip()

                    user_prompt = liquid_haskell_proof_user_prompt.format(
                        program_p=dedented_program_p,
                        program_q=dedented_program_q
                    )

                    alice_training_example = {
                        "system_prompt": liquid_haskell_proof_system_prompt,
                        "user_prompt": user_prompt,
                        "output": equiv_llm_response,
                    }

                    self.seq.cumulative_alice_training_data.append(alice_training_example)

                    return "proved", lh_process.stdout
                
                # 4. On failure, extract stderr and ask Alice for a new proof_body
                error_msg = lh_process.stderr
                logger.warning(f"❌ Liquid Haskell proof refuted (Attempt {attempt + 1}):\n{error_msg}")
                
                if attempt < MAX_PROOF_ATTEMPTS - 1:
                    logger.info("Asking Alice for a new proof body...")
                    proof_body, equiv_llm_response = self.generate_proof_body(
                        program_p, program_q, func_name_p, func_name_q, arg_type_p, error_msg, equiv_code_prompt_msg
                    )
                else:
                    logger.warning(f"Max proof attempts ({MAX_PROOF_ATTEMPTS}) reached. Final refutation.")
                    return "refuted", error_msg

            except subprocess.TimeoutExpired:
                logger.warning(f"Liquid Haskell verification timed out (Attempt {attempt + 1}).")
                # If it times out on the last attempt, return timeout
                if attempt == MAX_PROOF_ATTEMPTS - 1:
                    return "lh_timeout", "Liquid Haskell verification timed out"
            except Exception as e:
                logger.error(f"An unexpected error occurred during Liquid Haskell verification: {e}")
                return "lh_error", str(e)

        # This part should ideally not be reached if the loop handles all cases
        return "refuted", "Max proof attempts reached without success."

# --- SEQ Self-Play ---
class SEQ:
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name_or_path
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        torch.cuda.empty_cache()
        self._initialize_vllm()

        self.alice_adapter_path = args.alice_adapter_path
        self.bob_adapter_path = args.bob_adapter_path

        self.executor = CodeExecutor(self, timeout=args.timeout)
        self.programs = self.load_programs(args.dataset_name)
        
        self.cumulative_alice_training_data = self.load_cumulative_training_data(args.cumulative_alice_training_data_path)
        self.cumulative_bob_training_data = self.load_cumulative_training_data(args.cumulative_bob_training_data_path)


    def _initialize_vllm(self):
        logger.info("Initializing vLLM model...")
        self.vllm_model = vllm.LLM(
            model=self.model_name,
            tensor_parallel_size=self.args.tensor_parallel_size,
            trust_remote_code=True,
            enable_lora=True,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            max_model_len=self.args.max_tokens,
        )

        # print_gpu_memory_usage("After initializing vLLM model")
        print_nvidia_smi("After initializing vLLM model")
        print_gpu_memory_usage("After initializing vLLM model")

    def load_cumulative_training_data(self, programs_file_path):     
        if programs_file_path and os.path.exists(programs_file_path):
            logger.info(f"Loading programs from {programs_file_path}...")
            with open(programs_file_path, 'r') as f:
                return [json.loads(line.strip()) for line in f if line.strip()]
        return []

    def load_programs(self, dataset_name):

        logger.info(f"Loading initial programs from dataset {dataset_name}...")
        try:
            programs = []
            dataset_iterator = None

            if os.path.isdir(dataset_name):
                logger.info(f"Found local dataset directory. Loading from disk: {dataset_name}")
                dataset = load_from_disk(dataset_name)
                # Get the top N programs, ensuring not to go out of bounds
                num_to_take = self.args.num_initial_programs
                dataset = dataset.shuffle(seed=42)
                if num_to_take==0:
                    logger.info("`num_initial_programs` is not set, loading all programs.")
                    dataset_iterator = dataset
                else:
                    num_to_take = min(num_to_take, len(dataset))
                    logger.info(f"Initial programs: {len(dataset)}")
                    logger.info(f"Loading {num_to_take} initial programs.")
                    dataset_iterator = dataset.select(range(num_to_take))
            elif dataset_name.endswith('.jsonl'):
                logger.info(f"Found local .jsonl file. Loading from: {dataset_name}")
                dataset = load_dataset('json', data_files={'train': dataset_name}, split='train', streaming=True)
                dataset = dataset.shuffle(seed=42)
                if self.args.num_initial_programs==0:
                    logger.info("`num_initial_programs` is not set, loading all programs from stream.")
                    dataset_iterator = dataset
                else:
                    logger.info(f"Loading {self.args.num_initial_programs} initial programs from stream.")
                    dataset_iterator = dataset.take(self.args.num_initial_programs)
            else:
                logger.info(f"Loading from Hugging Face Hub: {dataset_name}")
                dataset = load_dataset(dataset_name, split='train', streaming=True)
                dataset = dataset.shuffle(seed=42)
                if self.args.num_initial_programs==0:
                    logger.info("`num_initial_programs` is not set, loading all programs from stream.")
                    dataset_iterator = dataset
                else:
                    logger.info(f"Loading {self.args.num_initial_programs} initial programs from stream.")
                    dataset_iterator = dataset.take(self.args.num_initial_programs)

            for program_item in dataset_iterator:
                if 'code' in program_item:
                    programs.append({'code': program_item['code']})

            if self.args.num_initial_programs!=0 and len(programs) < self.args.num_initial_programs:
                 logger.warning(f"Warning: Only able to load {len(programs)} programs, but {self.args.num_initial_programs} were requested.")

            logger.info(f"Loaded {len(programs)} programs from {dataset_name}.")
            return programs

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}. If using a local path, have you run the preprocessing script? Error: {e}")
            # Exit or return an empty list if dataset loading fails
            return []


    def parse_alice_equivalence_output(self, text):
        try:
            # print(f"Alice output: \n\n{text}\n\n")
            # Extract program Q
            program_q_match = re.search(r"\*\*Generated Program `?Q`?:\*\*\s*```haskell\n(.*?)\n```", text, re.DOTALL)
            if not program_q_match:
                logger.warning("🟥 --- Alice parsing failed: Could not find 'Generated Program Q' block ---")

                logger.warning(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            program_q = program_q_match.group(1).strip()
            
            if not program_q:
                logger.warning("🟥 --- Alice parsing failed: `Q` is empty after parsing ---")
                logger.warning(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            
            return program_q, text
        except Exception as e:
            logger.error(f"🟥 --- Alice parsing failed with exception: {e} ---")
            logger.error(f"Full output from Alice:\n{text}")
            return None, None
    
    def parse_alice_inequivalence_output(self, text):
        try:
            # logger.info(f"Alice output: \n\n{text}\n\n")
            # Extract program Q
            program_q_match = re.search(r"\*\*Generated Program `Q`:\*\*\s*```haskell\n(.*?)\n```", text, re.DOTALL)
            if not program_q_match:
                logger.error("🟥 --- Alice parsing failed: Could not find 'Generated Program Q' block ---")

                logger.error(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            program_q = program_q_match.group(1).strip()

            # Extract diverging input x
            diverging_input_match = re.search(r"\*\*Diverging Input `x`:\*\*\s*```(?:[^\n]*)\n(.*?)\n```", text, re.DOTALL)
            if not diverging_input_match:
                logger.error("🟥 --- Alice parsing failed: Could not find 'Diverging Input x' block ---")
                logger.error(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            diverging_input = diverging_input_match.group(1).strip()
            # Remove potential 'x = ' prefix from the diverging input.
            diverging_input = re.sub(r'^\s*\w+\s*=\s*', '', diverging_input).strip()
            
            if not program_q or not diverging_input:
                logger.error("🟥 --- Alice parsing failed: `Q` or `x` is empty after parsing ---")
                logger.error(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            
            return program_q, diverging_input
        except Exception as e:
            logger.error(f"🟥 --- Alice parsing failed with exception: {e} ---")
            logger.error(f"Full output from Alice:\n{text}")
            return None, None
        
    def run_alice_equivalence(self, program_p_completion):
        """Alice generates a variant of a program."""
        logger.info(f"Running Alice...")
        
        # Extract the argument type for the original program P
        arg_type_p = get_function_arg_type(program_p_completion)
        if not arg_type_p:
            logger.warning("Could not determine argument type for program P. Using 'a' as a placeholder.")
            arg_type_p = "a" # Fallback to a generic type variable

        # ALICE_EQUIV_USER_PROMPT already contains {program}
        user_content = ALICE_EQUIV_USER_PROMPT.format(
            difficulty_level=10,
            program_p_completion=program_p_completion,
            t=arg_type_p
        )
        
        messages = [
            {"role": "system", "content": ALICE_EQUIV_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        sampling_params = vllm.SamplingParams(
            n=1,
            temperature=self.args.temperature,
            max_tokens=self.args.max_tokens,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            presence_penalty=self.args.presence_penalty
        )
        
        # Note: vLLM currently doesn't support different LoRA adapters for different requests in a single batch.
        # We pass the LoRA request, and it will be applied to all prompts in the batch.
        

        for attempt in range(self.args.max_alice_retries):
            logger.info(f"Alice generation attempt {attempt + 1}/{self.args.max_alice_retries}...")
            request_outputs = self.vllm_model.generate(
                [prompt] * 1,
                sampling_params,
                lora_request=LoRARequest("alice_adapter", 1, self.alice_adapter_path) if self.alice_adapter_path else None
            )

            logger.info(f"Length of Alice request_outputs: {len(request_outputs)}")

            for output in request_outputs:
                result_text = output.outputs[0].text
                program_q, alice_raw_output = self.parse_alice_equivalence_output(result_text)
                
                if program_q:
                    # Check if the generated program Q compiles
                    if self.executor.check_compiles(program_q):
                        logger.info("Alice generated a compilable program Q.")
                        return program_q, alice_raw_output, user_content
                    else:
                        logger.warning("🟨 Alice's generated program Q failed to compile. Retrying...")
                else:
                    logger.warning("🟨 Alice failed to generate a valid program Q (parsing issue). Retrying...")
        
        logger.error(f"❌ Alice failed to generate a compilable program Q after {self.args.max_alice_retries} attempts.")
        return None, None, None

    def run_alice_inequivalence(self, program_p):
        """Alice generates a variant of a program."""
        logger.info(f"Running Alice...")
        
        user_content = ALICE_INEQUIV_USER_PROMPT.format(
            difficulty_level=10,
            program=program_p
        )
        messages = [
            {"role": "system", "content": ALICE_INEQUIV_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        sampling_params = vllm.SamplingParams(
            n=1,
            temperature=self.args.temperature,
            max_tokens=self.args.max_tokens,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            presence_penalty=self.args.presence_penalty
        )
        
        # Note: vLLM currently doesn't support different LoRA adapters for different requests in a single batch.
        # We pass the LoRA request, and it will be applied to all prompts in the batch.
        
        for attempt in range(self.args.max_alice_retries):
            logger.info(f"Alice generation attempt {attempt + 1}/{self.args.max_alice_retries}...")
            request_outputs = self.vllm_model.generate(
                [prompt],
                sampling_params,
                lora_request=LoRARequest("alice_adapter", 1, self.alice_adapter_path) if self.alice_adapter_path else None
            )

            logger.info(f"Length of Alice request_outputs: {len(request_outputs)}")

            # Assuming n=1 for Alice, so only one output to process
            output = request_outputs[0]
            result_text = output.outputs[0].text
            program_q, diverging_input = self.parse_alice_inequivalence_output(result_text)

            if program_q and diverging_input:
                # Before returning, check if the generated Q compiles
                if self.executor.check_compiles(program_q):
                    logger.info(f"Alice successfully generated a compilable program Q on attempt {attempt + 1}.")
                    return program_q, diverging_input, result_text
                else:
                    logger.warning(f"🟨 Alice generated a program Q that failed to compile on attempt {attempt + 1}. Retrying...")
            else:
                logger.warning(f"🟨 Alice parsing failed on attempt {attempt + 1}. Retrying...")
        
        logger.error(f"🟥 Alice failed to generate a compilable program after {self.args.max_alice_retries} attempts.")
        return None, None, None

    def call_alice_for_proof(self, proof_system_prompt: str, proof_user_prompt: str) -> str:
        """Generic method to call Alice with a specified prompt."""
        logger.info("Calling Alice for proof generation...")
        
        messages = [
            {"role": "system", "content": proof_system_prompt},
            {"role": "user", "content": proof_user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sampling_params = vllm.SamplingParams(
            n=1,
            temperature=self.args.temperature,
            max_tokens=self.args.max_tokens,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            presence_penalty=self.args.presence_penalty
        )
        
        request_outputs = self.vllm_model.generate(
            [prompt],
            sampling_params,
            lora_request=LoRARequest("alice_adapter", 1, self.alice_adapter_path) if self.alice_adapter_path else None
        )

        if request_outputs:
            return request_outputs[0].outputs[0].text
        return ""
    
    def parse_bob_output(self, text):
        try:
            # Extract the final output
            pattern = re.compile(
                    r'^#\s*Equivalent\?\s*'                    # the header
                    r'(?P<ans>yes|no|y|n)'                     # capture y/yes/n/no
                    r'(?:[.!?])?$',                            # allow an optional . ! or ?
                    re.IGNORECASE | re.MULTILINE
                )
            final_output_match = pattern.search(text)

            if final_output_match:
                tok = final_output_match.group('ans').lower()
                logger.info(f"Bob's output: {tok}")
                if tok == 'yes' or tok == 'y':
                    return True
                elif tok == 'no' or tok == 'n':
                    return False
                else:
                    logger.error(f"🟥 --- Bob parsing failed: Could not find 'Equivalent?' block ---")
                    logger.error(f"To solve this, please check the full output from Bob below and see why it failed to parse:\n{text}")
                    return None, None
        except Exception as e:
            logger.error(f"🟥 --- Bob parsing failed with exception: {e} ---")
            logger.error(f"Full output from Bob:\n{text}")
            return None, None
        
    def run_bob(self, program_p, program_q, alice_equivalence_status):
        logger.info(f"Running Bob to calculate difficulty over {self.args.n_samples} attempts...")
        
        messages = [
            {"role": "system", "content": BOB_SYSTEM_PROMPT},
            {"role": "user", "content": BOB_USER_PROMPT_TEMPLATE.format(program_p=program_p, program_q=program_q)}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        sampling_params = vllm.SamplingParams(
            n=self.args.n_samples,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            max_tokens=self.args.max_tokens,
            presence_penalty=self.args.presence_penalty
        )
        
        # We generate n_samples for a single prompt. This returns one RequestOutput.
        request_outputs = self.vllm_model.generate(
            [prompt],
            sampling_params,
            lora_request=LoRARequest("bob_adapter", 1, self.bob_adapter_path) if self.bob_adapter_path else None
        )

        bob_outputs = request_outputs[0].outputs
        n_correct = 0

        for output in bob_outputs:

            bob_output = output.text
            is_equivalent = self.parse_bob_output(bob_output)

            if (is_equivalent and alice_equivalence_status == "equivalent") or (not is_equivalent and alice_equivalence_status == "inequivalent"):
                bob_training_example = {
                    "system_prompt": BOB_SYSTEM_PROMPT,
                    "user_prompt": BOB_USER_PROMPT_TEMPLATE.format(program_p=program_p, program_q=program_q),
                    "output" : bob_output
                }
                self.cumulative_bob_training_data.append(bob_training_example)
                n_correct += 1
        
        difficulty = 10 * (1 - (n_correct / self.args.n_samples))
        logger.info(f"Bob's difficulty: {difficulty}")

        return difficulty

    def save_as_hf_jsonl(self, data: List[dict], file_path: str):
        # Detect if data is list of str, and wrap it
        if data and isinstance(data[0], str):
            records = [{"text": line} for line in data]
        else:
            records = data  # assume it's already List[dict]

        # 'data' is a list of dicts
        ds = Dataset.from_list(records)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        ds.to_json(file_path, orient="records", lines=True)
        logger.info(f"Saved HF JSONL dataset with {len(ds)} examples to {file_path}")

    def run_generation_iteration(self):
        logger.info("Starting data generation iteration...")

        summary_counts = {
            "p_compile_fail": 0,
            "alice_generation_fail": 0,
            "alice_identical_code": 0,
            "alice_not_divergent": 0,
            "Alice_generated_equivalent_program": 0,
            "Alice_generated_inequivalent_program": 0,
            "lh_refuted": 0,
            "lh_error": 0,
            "lh_timeout": 0,
        }

        # random.shuffle(self.programs)
        iteration_data = []
        candidate_alice_training_examples = []
        candidate_diff_pred_training_examples = []
        
        for j, program_item in enumerate(tqdm(self.programs, desc=f"Iteration {self.args.iteration}")):

            p_original_code = program_item['code']
            p_original_code = p_original_code.replace("main :: IO ()", "")
            p_original_code = strip_comments(p_original_code)


            tqdm.write(f"\n--- Processing example {j+1}/{len(self.programs)} in Iteration {self.args.iteration} ---")
            # tqdm.write(f"Original program (P):\n{p_original_code}")

            if not self.executor.check_compiles(p_original_code):
                logger.warning("🟨 Original program P failed to compile. Skipping example.")
                summary_counts["p_compile_fail"] += 1
                continue

            # Alice's turn
            if (random.random() < 0.5): # 50% chance to generate an equivalent program

                logger.info(f" *** Running Alice for equivalence *** ")

                q_candidate, alice_raw_output, alice_user_content = self.run_alice_equivalence(p_original_code)
            
                if not q_candidate:
                    logger.warning("🟨 Alice failed to generate a candidate program.")
                    summary_counts["alice_generation_fail"] += 1
                    continue

                logger.info(f"Alice's candidate program: \n{q_candidate}")
        
                # Phase 2: Embedding & Proving with Liquid Haskell
                lh_status, lh_output = self.executor.verify_equivalence(p_original_code, q_candidate)
                
                logger.info(f"Liquid Status: {lh_status}")
                logger.info(f"Liquid Output: {lh_output}")

                if lh_status == "proved":
                    logger.info("✅ Alice generated an equivalent program with a successful LH proof.")
                    summary_counts["Alice_generated_equivalent_program"] += 1
                    # Add to Alice's training data (proof-conditioned) 
                    bob_difficulty = self.run_bob(p_original_code, q_candidate, "equivalent")

                    iteration_data.append({
                        "p": p_original_code,
                        "q": q_candidate,
                        "Alice_goal": "equivalent",
                        "status": "proved",
                        "lh_output": lh_output,
                        "difficulty": bob_difficulty
                    })

                    diff_pred_user_content = ALICE_DIFFICULTY_PREDICTION_EQUIVALENT_SYSTEM_PROMPT_TEMPLATE.format(
                        program_p=p_original_code,
                        program_q=q_candidate
                    )
                    diff_pred_training_example = {
                        "system_prompt": diff_pred_user_content,
                        "user_prompt": ALICE_DIFFICULTY_PREDICTION_USER_PROMPT,
                        "output": f"Difficulty level: {bob_difficulty:.2f}",
                        "difficulty": bob_difficulty
                    }
                    candidate_diff_pred_training_examples.append(diff_pred_training_example)      

                    if bob_difficulty >= self.args.difficulty_threshold: # Alice is too smart to generate a program that Bob thinks it's inequivalent.
                        alice_training_example = {
                            "system_prompt": ALICE_EQUIV_SYSTEM_PROMPT,
                            "user_prompt": alice_user_content,
                            "output" : alice_raw_output,
                            "difficulty": bob_difficulty
                        }
                        candidate_alice_training_examples.append(alice_training_example)

                elif lh_status == "refuted":
                    logger.warning("❌ Alice's claim was refuted by Liquid Haskell.")
                    summary_counts["lh_refuted"] += 1 
                    continue
                    
                elif lh_status == "lh_timeout":
                    logger.warning("⏱️ Liquid Haskell verification timed out. Skipping example.")
                    summary_counts["lh_timeout"] += 1
                    # Do NOT append to iteration_data or training data, as it's an unreliable example.
                    continue
                elif lh_status == "identical":
                    logger.warning("🟨 Alice generated an identical program Q to P. Skipping example.")
                    summary_counts["alice_identical_code"] += 1
                    continue
                else: # lh_error
                    logger.error("🔴 Liquid Haskell verification failed unexpectedly. Skipping example.")
                    summary_counts["lh_error"] += 1
                    # Do NOT append to iteration_data or training data.
                    continue

            else: # 50% chance to generate an inequivalent program

                logger.info(f" *** Running Alice for inequivalence *** ")
                q_candidate, x_candidate, alice_raw_output = self.run_alice_inequivalence(p_original_code)
            
                if not q_candidate:
                    logger.warning("🟨 Alice failed to generate a candidate program.")
                    summary_counts["alice_generation_fail"] += 1
                    continue

                is_divergent_alice = self.executor.check_divergence(p_original_code, q_candidate, x_candidate)
                
                if not is_divergent_alice:
                    logger.warning("🟨 Alice's candidate program and diverging input were not divergent / Execution failed. Skipping.")
                    summary_counts["alice_not_divergent"] += 1
                    continue

                logger.info(f"Alice's candidate program: \n{q_candidate}")
                logger.info(f"Alice's diverging input: {x_candidate}")
                
                summary_counts["Alice_generated_inequivalent_program"] += 1
                bob_difficulty = self.run_bob(p_original_code, q_candidate, "inequivalent")

                iteration_data.append({
                        "p": p_original_code,
                        "q": q_candidate,
                        "Alice_goal": "inequivalent",
                        "status": "inequivalent",
                        "diverging_input": x_candidate,
                        "difficulty": bob_difficulty
                    })
                
                diff_pred_user_content = ALICE_DIFFICULTY_PREDICTION_INEQUIVALENT_SYSTEM_PROMPT_TEMPLATE.format(
                    program_p=p_original_code,
                    program_q=q_candidate
                )

                diff_pred_training_example = {
                    "system_prompt": diff_pred_user_content, 
                    "user_prompt": ALICE_DIFFICULTY_PREDICTION_USER_PROMPT,
                    "output": f"Difficulty level: {bob_difficulty:.2f}",
                    "difficulty": bob_difficulty
                }
                candidate_diff_pred_training_examples.append(diff_pred_training_example)
                    
                if bob_difficulty >= self.args.difficulty_threshold:
                    alice_training_example = {
                        "system_prompt": ALICE_INEQUIV_SYSTEM_PROMPT,
                        "user_prompt": ALICE_INEQUIV_USER_PROMPT.format(difficulty_level=bob_difficulty, program=p_original_code),
                        "output" : alice_raw_output,
                        "difficulty": bob_difficulty
                    }
                    candidate_alice_training_examples.append(alice_training_example)

        # --- Task 3: Biasing Alice's Training Set --- 
        hard_examples = [ex for ex in candidate_alice_training_examples if ex["difficulty"] >= self.args.difficulty_threshold]
        easy_examples = [ex for ex in candidate_alice_training_examples if ex["difficulty"] < self.args.difficulty_threshold]
        
        logger.info(f"Number of hard examples: {len(hard_examples)}")
        logger.info(f"Number of easy examples: {len(easy_examples)}")
        
        # Group easy examples by integer difficulty bins
        num_easy_bins = int(self.args.difficulty_threshold) if self.args.difficulty_threshold >= 1 else 1
        bins = {k: [] for k in range(num_easy_bins)}
        for ex in easy_examples:
            bin_idx = int(ex["difficulty"])
            if bin_idx < num_easy_bins:
                bins[bin_idx].append(ex)

        # Sample 20% of the number of hard examples from the easy examples
        needed = int(0.2 * len(hard_examples))
        sampled_easies = []
        
        # Round-robin sampling from bins
        if easy_examples:
            i = 0
            while len(sampled_easies) < needed:
                bin_idx = i % num_easy_bins
                if bins[bin_idx]:
                    sampled_easies.append(bins[bin_idx].pop(random.randrange(len(bins[bin_idx]))))
                i += 1
                # Break if we have cycled through all bins and can't find any more examples
                if i > 0 and i % num_easy_bins == 0:
                    if sum(len(b) for b in bins.values()) == 0:
                        break
        
        final_alice_examples = hard_examples + sampled_easies
        
        logger.info(f"Number of candidate alice examples: {len(candidate_alice_training_examples)}")
        logger.info(f"Number of final alice examples: {len(final_alice_examples)}")
        logger.info(f"Number of hard examples: {len(hard_examples)}")
        logger.info(f"Number of easy examples: {len(sampled_easies)}")
        

        for ex in final_alice_examples:
            alice_training_example = {
                "system_prompt": ex["system_prompt"],
                "user_prompt": ex["user_prompt"],
                "output": ex["output"]
            }
            self.cumulative_alice_training_data.append(alice_training_example)

        # Biasing for diff_pred_training_examples
        hard_diff_pred_examples = [ex for ex in candidate_diff_pred_training_examples if ex["difficulty"] >= self.args.difficulty_threshold]
        easy_diff_pred_examples = [ex for ex in candidate_diff_pred_training_examples if ex["difficulty"] < self.args.difficulty_threshold]

        logger.info(f"Number of hard diff pred examples: {len(hard_diff_pred_examples)}")
        logger.info(f"Number of easy diff pred examples: {len(easy_diff_pred_examples)}")
        
        num_hard_diff_pred_examples = len(hard_diff_pred_examples)
        needed_easy_diff_pred_examples_count = int(0.5 * num_hard_diff_pred_examples)
        
        sampled_easy_diff_pred_examples = []
        if easy_diff_pred_examples:
            random.shuffle(easy_diff_pred_examples) # Shuffle to ensure random sampling
            sampled_easy_diff_pred_examples = easy_diff_pred_examples[:needed_easy_diff_pred_examples_count]

        final_diff_pred_examples = hard_diff_pred_examples + sampled_easy_diff_pred_examples

        logger.info(f"Number of candidate diff pred examples: {len(candidate_diff_pred_training_examples)}")    
        logger.info(f"Number of final diff pred examples: {len(final_diff_pred_examples)}")
        logger.info(f"Number of hard diff pred examples: {len(hard_diff_pred_examples)}")
        logger.info(f"Number of easy diff pred examples: {len(sampled_easy_diff_pred_examples)}")
        

        for ex in final_diff_pred_examples:
            diff_pred_training_example = {
                "system_prompt": ex["system_prompt"],
                "user_prompt": ex["user_prompt"],
                "output": ex["output"]
            }
            self.cumulative_alice_training_data.append(diff_pred_training_example)
            
        # Save iteration data to JSON
        iter_output_dir = self.args.iteration_dir
        if iteration_data: 
            iteration_data_path = os.path.join(iter_output_dir, "iteration_data.jsonl")
            self.save_as_hf_jsonl(iteration_data, iteration_data_path)

        # Also save the training data for finetuning
        alice_training_path = os.path.join(iter_output_dir, "alice_training_data.jsonl")
        self.save_as_hf_jsonl(self.cumulative_alice_training_data, alice_training_path)
        
        if self.cumulative_bob_training_data:
            bob_training_path = os.path.join(iter_output_dir, "bob_training_data.jsonl")
            self.save_as_hf_jsonl(self.cumulative_bob_training_data, bob_training_path)

        # Log summary statistics
        logger.info("\n--- Iteration Summary ---")
        logger.info(f"Total programs processed: {len(self.programs)}")
        logger.info("Summary Counts:")
        for summary_type, count in summary_counts.items():
            logger.info(f"  {summary_type}: {count}")

        if iteration_data: 
            avg_diff = sum(item['difficulty'] for item in iteration_data) / len(iteration_data)
            logger.info(f"Average Difficulty: {avg_diff}")
            logger.info(f"Number of examples with difficulty > {self.args.difficulty_threshold}: {sum(1 for item in iteration_data if item['difficulty'] > self.args.difficulty_threshold)}")
            logger.info(f"Number of examples with difficulty < {self.args.difficulty_threshold}: {sum(1 for item in iteration_data if item['difficulty'] < self.args.difficulty_threshold)}")

        # Save summary statistics to a separate JSONL file
        summary_data = [
            {
                "total_programs_processed": len(self.programs),
                "summary_counts": summary_counts,
                "avg_difficulty": avg_diff if iteration_data else None,
                "accuracy": (10 - avg_diff) * 10 if iteration_data else None,
                f"Number of examples with difficulty > {self.args.difficulty_threshold}": sum(1 for item in iteration_data if item['difficulty'] > self.args.difficulty_threshold),
                f"Number of examples with difficulty < {self.args.difficulty_threshold}": sum(1 for item in iteration_data if item['difficulty'] < self.args.difficulty_threshold)
            }
        ]
        summary_file_path = os.path.join(iter_output_dir, "seq_summary.jsonl")
        self.save_as_hf_jsonl(summary_data, summary_file_path)

        torch.cuda.empty_cache()
        logger.info("Generation iteration complete.")


def main():
    parser = argparse.ArgumentParser(description="SEQ Data Generation")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the base model.")

    parser.add_argument('--dataset_name', type=str, default='../data/successfully_compiled_sorted_haskell_dataset', help="Hugging Face dataset name or local path for initial programs.")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save models and results.")
    parser.add_argument('--max_tokens', type=int, default=32768, help="Maximum number of tokens for the model's context window and generation.")
    parser.add_argument('--temperature', type=float, default=0.6, help="Temperature for sampling.")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p for sampling.")
    parser.add_argument('--top_k', type=int, default=20, help="Top-k for sampling.")
    parser.add_argument('--presence_penalty', type=float, default=1.5, help="Presence penalty for sampling.")
    parser.add_argument('--timeout', type=float, default=60.0, help="Timeout for code execution.")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help="The fraction of GPU memory to be used for the vLLM KV cache.")
    parser.add_argument('--num_initial_programs', type=int, default=100, help="Number of initial programs to load.")
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="Number of tensor parallel processes.")
    parser.add_argument('--n_samples', type=int, default=10, help="Number of samples to generate for Bob.")
    parser.add_argument('--difficulty_threshold', type=float, default=5, help="Difficulty threshold for Bob to be considered successful.")
    parser.add_argument('--max_alice_retries', type=int, default=3, help="Maximum number of retries for Alice to generate a program.")
    
    parser.add_argument('--iteration', type=int, required=True, help="Current self-play iteration number.")
    parser.add_argument('--iteration_dir', type=str, required=True, help="Path to the directory for the current iteration.")
    parser.add_argument('--alice_adapter_path', type=str, default=None, help="Path to an adapter for Alice.")
    parser.add_argument('--bob_adapter_path', type=str, default=None, help="Path to an adapter for Bob.")
    parser.add_argument('--cumulative_alice_training_data_path', type=str, default=None, help="Path to a file with the list of programs for the current iteration.")
    parser.add_argument('--cumulative_bob_training_data_path', type=str, default=None, help="Path to a file with the list of programs for the current iteration.")

    args = parser.parse_args()

    random.seed(42)

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    seq = SEQ(args)

    try:
        seq.run_generation_iteration()
    finally:
        # Explicitly clean up the temporary directory created by CodeExecutor
        if hasattr(seq, 'executor') and seq.executor and os.path.exists(seq.executor.tmp_dir):
            logger.info(f"Cleaning up temporary directory: {seq.executor.tmp_dir}")
            shutil.rmtree(seq.executor.tmp_dir)

if __name__ == "__main__":
    main()



