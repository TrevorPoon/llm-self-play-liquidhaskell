
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_results(results_dirs):
    all_data = []
    for results_dir in results_dirs:
        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(results_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                    data["model_type"] = "finetuned" if data["setup"]["adapter_path"] else "base"
                    all_data.append(data)
    return pd.DataFrame(all_data)

def plot_metrics(df):
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    configs = df["datasets"].unique()

    sns.set_theme(style="whitegrid", palette="deep")

    for metric in metrics:
        plt.figure(figsize=(12, 7))
        
        # Prepare data for plotting
        plot_df = pd.DataFrame()
        for cfg in configs:
            for model_type in ["base", "finetuned"]:
                subset = df[(df["datasets"] == cfg) & (df["model_type"] == model_type)]
                if not subset.empty:
                    # For simplicity, let's take the mean if there are multiple iterations/runs
                    # You might want to plot all individual points or show variance depending on data
                    mean_metric = subset["evaluation_metrics"].apply(lambda x: x.get(metric, 0)).mean() * 100
                    plot_df = pd.concat([
                        plot_df,
                        pd.DataFrame([{"config": cfg, "model_type": model_type, "value": mean_metric}])
                    ])

        if plot_df.empty:
            print(f"No data to plot for metric: {metric}")
            continue

        sns.barplot(x="config", y="value", hue="model_type", data=plot_df, palette={"base": "skyblue", "finetuned": "lightcoral"})
        for container in plt.gca().containers:
            plt.bar_label(container, fmt='%.1f%%', fontsize=10)
        plt.title(f"Comparison of {metric.replace('_', ' ').title()} Across Datasets", fontsize=16)
        plt.xlabel("Dataset Configuration", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.legend(title="Model Type")
        plt.ylim(0, 100) # Metrics are usually between 0 and 100
        plt.tight_layout()
        plt.savefig(f"{metric}_comparison_EquiBench.pdf")
        plt.close()
        print(f"Generated figures/{metric}_comparison_EquiBench.pdf")

    # Print confusion matrices
    print("\n--- Confusion Matrices ---")
    for cfg in configs:
        for model_type in ["base", "finetuned"]:
            subset = df[(df["datasets"] == cfg) & (df["model_type"] == model_type)]
            if not subset.empty:
                # Sum confusion matrices element-wise and then average
                sum_confusion_matrix = [[0, 0], [0, 0]]
                count = 0
                for _, row in subset.iterrows():
                    cm = row["evaluation_metrics"].get("confusion_matrix", [[0,0],[0,0]])
                    if cm:
                        sum_confusion_matrix[0][0] += cm[0][0]
                        sum_confusion_matrix[0][1] += cm[0][1]
                        sum_confusion_matrix[1][0] += cm[1][0]
                        sum_confusion_matrix[1][1] += cm[1][1]
                        count += 1
                    unparsed_samples = row["predictions_summary"].get("unparsed_samples")
                
                if count > 0:
                    avg_confusion_matrix = [[int(val / count) for val in row] for row in sum_confusion_matrix]
                    print(f"\nDataset: {cfg}, Model Type: {model_type}")
                    print(pd.DataFrame(avg_confusion_matrix, index=["True Negative", "True Positive"], columns=["Predicted Negative", "Predicted Positive"]))
                    print("Unparsed samples: ", unparsed_samples)

if __name__ == "__main__":
    results_dirs = ["../base", os.path.dirname(os.path.abspath(__file__))]
    
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    df = load_results(results_dirs)
    if not df.empty:
        plot_metrics(df)
    else:
        print("No evaluation results found to plot.")

