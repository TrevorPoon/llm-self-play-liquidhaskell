
## 1. Injecting Target Difficulty into Alice’s Prompt

### 1.1 Why

Alice must know “how hard” we want her next challenge to be.  In the paper, at generation time we always feed her a **target difficulty** $d_t$ (initially 10), so she learns to calibrate how subtle her Q/x pair should be.

### 1.2 What to change in your code

* **Prompt template**
  Replace your current `ALICE_USER_PROMPT_TEMPLATE` (which only contains the original program) with something like:

  ````python
  ALICE_USER_PROMPT = textwrap.dedent("""
    Difficulty level: {difficulty_level}
    Original program `P`:
    ```haskell
    {program}
    ```
  ````

  """).strip()

  ```
  ```
* **At generation time**
  When you call `run_alice(p_original, dt)`, build the user message as:

  ```python
  user_content = ALICE_USER_PROMPT.format(
    difficulty_level=dt,              # e.g. 10 or measured d from last round
    program=p_original
  )
  ```
* **System message stays the same** (your `ALICE_SYSTEM_PROMPT`), so the model sees both the “game rules” and the *difficulty*.

---

## 2. Constructing Two Kinds of SFT Examples for Alice

The paper (§2.4) creates **(A) main** examples that teach Alice to hit a specific difficulty, and **(B) difficulty-prediction** examples so Alice learns to estimate how hard her own instances are.

### 2.1 (A) Main SFT Examples

1. **Generate**

   * Sample outputs from Alice with target $d_t$.
   * Parse each into $(\text{CoT}, Q, x)$.

2. **Measure**

   * For each candidate $(P,Q,x)$, run Bob $N$ times, count $N_{\text{correct}}$, compute

     $$
       d = 10 \times\bigl(1 - \tfrac{N_{\text{correct}}}{N}\bigr).
     $$

3. **Pack the example**

   * **Input** (user turn) = exactly the same template you used for generation, but replace the “Difficulty level: {d\_t}” with:

     ````text
     Difficulty level: {measured_d}
     ```haskell
     {P}
     ````
   * **Output** (assistant turn) = the raw text you got from Alice (CoT + Q + x).
   * **System prompt** remains the same.

4. **Add to Alice’s SFT buffer**

   ```python
   alice_main_example = {
     "system": ALICE_SYSTEM_PROMPT,
     "user": user_content_with_measured_d,
     "assistant": alice_raw_output
   }
   ```

### 2.2 (B) Difficulty-Prediction Examples

1. **Input #1**

   ````text
   Difficulty level: Any
   ```haskell
   {P}
   ````

   **Output #1** = the same Alice raw output (CoT+Q+x)
   *(we do *not* compute loss on these tokens)*

2. **Input #2**

   ```text
   Predict the difficulty level of the above instance.
   ```

   **Output #2**

   ```text
   Difficulty level: {measured_d}
   ```

   *(we compute cross-entropy only on this second assistant message)*

3. **Pack into your dataset**

   ```python
   alice_diff_pred_example = [
     {"role":"system","content":ALICE_SYSTEM_PROMPT},
     {"role":"user","content": prompt_any},
     {"role":"assistant","content": alice_raw_output},
     {"role":"user","content":"Predict the difficulty level of the above instance."},
     {"role":"assistant","content":f"Difficulty level: {measured_d}"}
   ]
   ```

---

## 3. Biasing Your Alice-Training Set

Left unchecked, easy examples will vastly outnumber the hard ones and drown out the learning signal.  The paper (§2.4) prescribes:

1. **Partition** your main examples by difficulty:

   * **Hard**: $d \ge 5$
   * **Easy**: $d < 5$

2. **Keep**

   * **All** of the hard examples
   * **Only 20%** of the easy ones, selected by **round-robin** across the integer difficulty bins $d\in [0,1), [1,2), …, [4,5)$.

   ```python
   hard = [e for e in main_examples if e.d >= 5]
   easy = [e for e in main_examples if e.d < 5]

   # group easies by floor(d)
   bins = {i:[] for i in range(5)}
   for e in easy:
     bins[int(e.d)].append(e)

   # sample 20% of total_hard size, cycling through bins
   needed = int(0.2 * len(hard))
   sampled_easies = []
   i = 0
   while len(sampled_easies) < needed:
     bin_list = bins[i % 5]
     if bin_list:
       sampled_easies.append(bin_list.pop(random.randrange(len(bin_list))))
     i += 1

   final_alice_training = hard + sampled_easies
   ```

3. **Combine**

   * All of `final_alice_training` **plus** the matching difficulty-prediction examples (you can bias those 50/50 hard/easy but loss only on prediction).

---

## 4. Finetuning from the base

In the SInQ self-play setup, each round of adapter training for Alice is begun from the original base checkpoint, not from the adapter produced in the previous round. Concretely:

“Each of Alice’s training runs starts from the baseline rather than the previous fine-tuned checkpoint, but we accumulate instances to be used as training examples between rounds.” 

The rationale is that, by always re-initializing from the same base model and re-training on the cumulative pool of examples (including newly generated hard cases), you avoid:

Bias compounding – any quirks or overfitting in one adapter won’t be baked into the next.

Stale difficulty estimates – since Bob remains unchanged throughout Alice’s rounds, you can trust the original difficulty labels without re-computing them.

If instead you fine-tuned “on top of” the previous adapter, you’d risk reinforcing early-round patterns (and potentially veering away from optimal parameter space), and you’d have to carefully manage optimizer states to prevent catastrophic forgetting.

So: start from scratch each time, but feed in all past and current examples when you launch that round of adapter training.
