# Semantic Inequivalence Game (SInQ) Self-Play

This directory contains the implementation of a Self-Play framework for the Semantic Inequivalence Game (SInQ), as described in the research paper. The goal of this framework is to train large language models (LLMs) to generate semantically inequivalent programs and their corresponding diverging inputs.

## Core Components

The SInQ game involves two primary agents, Alice and Bob, and an execution environment:

### 1. Alice (The Generator)
Alice's role is to generate a program `Q` that is semantically inequivalent to a given original program `P`. Crucially, Alice must also provide a 'diverging input' `x`. This input `x` should be valid for both `P` and `Q`, but when executed with `x`, `P` and `Q` must produce different outputs or behaviors (e.g., one halts and the other does not).

Alice's generations are designed to be subtly different, making it challenging for an evaluator to find the diverging input.

### 2. Bob (The Evaluator)
Bob's task is to determine if two given programs (`P` and `Q`) are semantically equivalent. If Bob determines they are inequivalent, it must provide a diverging input `x` that demonstrates this difference. Bob's ability to find such inputs is used to calculate a 'difficulty' score for the generated `(P, Q, x)` triplet.

### 3. Code Executor
The `CodeExecutor` (implemented in `src/Self-Play/SINQ/run_blastwind_without_diff.py`) is responsible for compiling and executing Haskell programs with given inputs. It verifies if two programs indeed diverge on a specific input, which is crucial for validating Alice's outputs and Bob's proposed diverging inputs.

## The Self-Play Loop

The SInQ framework operates through an iterative self-play loop:

1.  **Initial Programs**: The process starts with a set of initial Haskell programs (P). These are loaded from a dataset.

2.  **Alice Generates**: For each program `P` from the current set, Alice generates a candidate inequivalent program `Q` and a diverging input `x`.

3.  **Bob Evaluates**: Bob then attempts to find diverging inputs for the `(P, Q)` pair generated by Alice. The number of successful diverging inputs Bob finds (out of `n_samples`) determines the 'difficulty' of Alice's generated instance. A higher difficulty means Bob struggled more to find a diverging input.

4.  **Training Data Collection**: 
    *   **Alice's Training**: Alice's training data is constructed from the `(P, Q, x)` pairs she generates, along with the 'difficulty' score assigned by Bob. The training set is biased: all 'hard' examples (where Bob struggled) are kept, while only a small percentage of 'easy' examples are sampled to ensure Alice focuses on generating challenging instances.
    *   **Bob's Training**: Bob's training data consists of instances where he successfully found a diverging input. If Bob fails to find any input, a hard training example is created for him using Alice's original diverging input.

5.  **Model Fine-tuning**: After each iteration, both Alice's and Bob's underlying LLM adapters are fine-tuned using the newly collected training data. The key aspect of this setup, as implemented in `run_blastwind_without_diff.py`, is that **Alice's training always starts from the base model, accumulating instances over rounds**, rather than continuing from the previous adapter. This prevents bias compounding and ensures consistent difficulty estimation.

6.  **Program Expansion**: The `Q` programs generated by Alice that are deemed 'hard' (i.e., Bob found them difficult) are added to the pool of programs `P` for the next iteration, increasing the complexity and diversity of the problem set.

## Without Difficulty Prediction

The `run_blastwind_without_diff.py` script specifically implements a version of the SInQ game where Alice is *not* explicitly prompted with a target difficulty during generation. Instead, Alice's training data is simply filtered based on the difficulty score Bob assigns, without her needing to predict or aim for a specific difficulty level. This simplifies Alice's task to purely generating inequivalent programs and diverging inputs, with the difficulty aspect handled implicitly through data biasing.

## Running the Experiment

The `finetune_blastwind_pgr_without_diff.sh` script is designed to execute the self-play loop using the `run_blastwind_without_diff.py` script. It handles environment setup, resource allocation (e.g., GPU usage with SLURM), and passes necessary arguments to the Python script to configure the model, dataset, output directories, and self-play parameters.

**To run the self-play experiment, execute the `finetune_blastwind_pgr_without_diff.sh` script.**

This framework allows for continuous improvement of LLMs in understanding semantic equivalence and generating targeted program variants, which has implications for automated program analysis, testing, and vulnerability discovery. 