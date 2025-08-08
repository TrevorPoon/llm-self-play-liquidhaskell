import json
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_difficulty_chart(base_path, output_folder):
    iterations_data = []
    
    # Collect data for each iteration
    i = 1
    while True:
        iteration_path = os.path.join(base_path, f"iteration_{i}")
        if not os.path.isdir(iteration_path):
            break
        
        candidate_examples_path = os.path.join(iteration_path, "iteration_data.jsonl")
        
        difficulties = []
        if os.path.exists(candidate_examples_path):
            with open(candidate_examples_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "difficulty" in data:
                            difficulties.append(data["difficulty"])
                    except json.JSONDecodeError:
                        continue
        
        if difficulties:
            iterations_data.append({
                "round": f"Round {i}",
                "mean_difficulty": np.mean(difficulties),
                "std_deviation": np.std(difficulties)
            })
        i += 1

    if not iterations_data:
        print("No difficulty data found in any iteration.")
        return

    # Prepare data for plotting
    rounds = [data["round"] for data in iterations_data]
    mean_difficulties = [data["mean_difficulty"] for data in iterations_data]
    std_deviations = [data["std_deviation"] for data in iterations_data]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, mean_difficulties, marker='o', label='Mean difficulty', color='blue')
    plt.fill_between(rounds, np.array(mean_difficulties) - np.array(std_deviations),
                     np.array(mean_difficulties) + np.array(std_deviations),
                     color='blue', alpha=0.2, label='Standard Deviation')

    plt.title("Mean and Sd. difficulties for Each Alice's round (Bob's generation 1)")
    plt.xlabel("Alice's training round")
    plt.ylabel("Difficulty")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 5)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_image_path = os.path.join(output_folder, "difficulty_chart.png")
    plt.savefig(output_image_path)
    print(f"Chart saved to {output_image_path}")

# Example usage:
# Adjust the base_output_dir to your specific path
base_output_dir = "output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_LR2e-4_EPOCHS3"
output_chart_dir = base_output_dir

generate_difficulty_chart(base_output_dir, output_chart_dir)
