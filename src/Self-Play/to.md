
hs vllm base run -- manual review

blastwind/github-code-haskell-file -- preprocess on function? 

Create a python script and bash  in /home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SINQ/utils/execution.py
/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SINQ/utils/utils.py with reference to @run_blastwind_without_diff.py @finetune_blastwind_pgr_without_diff.sh and @execution.py @utils.py . I want it to use TheBloke/deepseek-coder-33B-instruct-GGUF in vllm to ask it to generate an input for each haskell code from the dataset , and then i want you to try compile and execute it to see if the input is good or not. Save all the dataset that is compilable and executive to a new dataset. Please use the most efficient way to implement the above process.

# Evaluation
livecodebench
MBPP

#SFT
Synthetic Dataset
lr and epochs experiment

#SHQ
remove check compile
Use the SFT adapter and then finetune on it 
batch size experiment
lr experiment


