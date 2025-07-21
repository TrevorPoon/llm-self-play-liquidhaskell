
# Evaluation

hs vllm manual review errors -- 📌 TODO

# SFT

Fails -- overfitting to haskell output

# SINQ

## BLASTWIND

Too noisy

## synthetic openinstruct

Generation dataset (10,000) --  ✅ Done 

Filter it with execution of random input --  ✅ Done

SINQ on synthetic -- ⏳ Running

Evaluation -- 📌 TODO

## Misc

run with difficulty -- 📌 TODO

Generation dataset (200,000) -- 📌 TODO 

Inspect the code carefully. Gimme me suggestions / improve / identify any bugs in my semantic equivalent game for LLM code generation -- 📌 TODO 

## Result

Blastwind (validate with function binding) -- Running Bob at 14% --  ✅ Done

Blastwind (validate with input running) -- Running Bob at 20% --  ✅ Done

OpenInstruct -- 📌 TODO

# SEQ

## synthetic openinstruct

test if deepseekcoder has liquidhaskell ability --  ✅ Done

Finetune on Alice Liquid-Haskell proof ⏳ Running

Ask how to prove, via liquid haskell and do i need Bob

Verification Phase (LLM Re-Prompt)
• If proof passed, prompt LLM for a human-style proof sketch in comments.
• If proof failed, prompt LLM for a concrete counterexample input.
• Collect:
– Positive example: (spec, original, variant, proof sketch).
– Negative example: (spec, original, variant, counterexample).

SEQ on synthetic -- 📌 TODO

Evaluation -- 📌 TODO



