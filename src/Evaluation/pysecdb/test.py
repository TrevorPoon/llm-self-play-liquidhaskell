from datasets import load_dataset

dataset = load_dataset("sunlab/PySecDB")

print(dataset["train"][0])