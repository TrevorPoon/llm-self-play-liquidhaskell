# LiveCodeBench - Code Generation Lite Evaluation

This guide provides instructions on how to set up and run the LiveCodeBench code generation evaluation. 

## Offline Version

To run the evaluation in offline mode, follow these steps:

1.  **Clone the Dataset:** First, clone the `code_generation_lite` dataset from Hugging Face:
    ```bash
    git clone https://huggingface.co/datasets/livecodebench/code_generation_lite
    ```

2.  **Execute the Evaluation Script:** Navigate into the cloned `code_generation_lite` folder and run the `eval_livecodebench.sh` script with offline environment variables set:
    ```bash
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    ./code_generation_lite/eval_livecodebench.sh
    ```

## Online Version

To run the evaluation in online mode, simply execute the `eval_livecodebench.sh` script directly:

```bash
./eval_livecodebench.sh
```