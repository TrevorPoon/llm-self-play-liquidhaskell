Step 1: Sort dataset (Using size as a proxy for code difficulty)
```
python sort_dataset.py
```

Step 2: validate dataset (Filter those that needs an input, function binding and compilation check)
```
python validate_dataset.py
```

Step 3: runs the function with actual value and check that read can parse that value
```
python generate_haskell_inputs.py 
```

Recommended 3-tier data strategy
Make tiers, don’t filter destructively:

Tier	Name	Criteria	Use in training / self-play
A	Raw	All raw code (after light de-dupe / parse).	Pretraining, code infilling, syntax repair tasks. High recall.
B	Typechecked / Needs-Input	Passes validator.py (has sig, binding, input arrow, GHC -fno-code).	Self-play theory phase: reason about types, generate candidate inputs symbolically.
C	Runnable Sample	Subset of B where input_generator.py produced a value that compiled & ran under harness.	High-trust evaluation, automated semantic inequivalence scoring, RL signal.


