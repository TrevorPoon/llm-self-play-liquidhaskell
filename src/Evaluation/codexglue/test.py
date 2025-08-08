from datasets import load_dataset

dataset = load_dataset("google/code_x_glue_cc_defect_detection", split="test")

print(dataset[0])