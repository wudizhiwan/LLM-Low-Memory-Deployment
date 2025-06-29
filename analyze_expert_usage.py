import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def analyze_expert_continuity(file_path: str, target_layer: str = "layer_0"):
    """
    Analyzes the continuity of expert usage in a MOE model from a given JSON file.

    Args:
        file_path (str): Path to the expert_usage JSON file.
        target_layer (str): The target layer to analyze, e.g., 'layer_0'.
    """
    print(f"Analyzing file: {file_path}")
    print(f"Target layer: {target_layer}")

    # --- 1. Load and preprocess data ---
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found -> {file_path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter for generated tokens (non-input)
    generated_tokens = [token for token in data.get("tokens", []) if not token.get("is_input")]
    if not generated_tokens:
        print("Error: No generated token data found in the JSON file.")
        return

    # Extract expert usage for the target layer
    layer_expert_usage = []
    for token in generated_tokens:
        usage = token.get("experts_usage", {}).get(target_layer)
        if usage is not None:
            layer_expert_usage.append({
                "token_text": token.get("token_text", "").replace(" ", "_"), # Replace space for better display
                "experts": usage
            })
    
    if not layer_expert_usage:
        print(f"Error: No expert usage data found for '{target_layer}' in generated tokens.")
        return
        
    # Get the total number of experts
    try:
        num_experts = data["overall_stats"]["per_layer_stats"][target_layer]["total_experts"]
    except KeyError:
        print("Warning: Could not determine total number of experts from metadata. Inferring from data.")
        all_used_experts = set()
        for usage in layer_expert_usage:
            all_used_experts.update(usage["experts"])
        if not all_used_experts:
            print("Error: Expert list is empty. Cannot continue analysis.")
            return
        num_experts = max(all_used_experts) + 1

    print(f"Found {len(layer_expert_usage)} generated tokens. Total experts in model: {num_experts}")

    # --- 2. Prepare data matrix for visualization ---
    # Rows are tokens, columns are experts. Value is 1 if the expert was activated.
    df_data = []
    token_labels = []
    for i, usage in enumerate(layer_expert_usage):
        token_labels.append(f"{i}")
        expert_activation = [0] * num_experts
        for expert_idx in usage["experts"]:
            if expert_idx < num_experts:
                expert_activation[expert_idx] = 1
        df_data.append(expert_activation)
    
    df = pd.DataFrame(df_data, index=token_labels, columns=[f"E{i}" for i in range(num_experts)])

    # --- 3. Visualization: Expert Usage Heatmap ---
    plt.style.use('default')
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(df.T, cmap="YlGnBu", cbar=False, linewidths=.5)
    plt.title(f"Expert Activation per Token for {target_layer}", fontsize=16)
    plt.xlabel("Generated Token (Index)", fontsize=12)
    plt.ylabel("Expert Index", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_filename_heatmap = f"expert_continuity_{target_layer}.png"
    plt.savefig(output_filename_heatmap)
    print(f"\nContinuity heatmap saved to: {output_filename_heatmap}")
    plt.show()


    # --- 4. Quantitative Analysis: Expert Overlap between Consecutive Tokens ---
    jaccard_scores = []
    for i in range(1, len(layer_expert_usage)):
        prev_experts = set(layer_expert_usage[i-1]["experts"])
        current_experts = set(layer_expert_usage[i]["experts"])
        
        intersection = len(prev_experts.intersection(current_experts))
        union = len(prev_experts.union(current_experts))
        
        if union == 0:
            jaccard_scores.append(0)
        else:
            jaccard_scores.append(intersection / union)

    if jaccard_scores:
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
        print(f"\nQuantitative Analysis:")
        print(f"  - Average expert overlap between consecutive tokens (Jaccard Similarity): {avg_jaccard:.4f}")
        if avg_jaccard > 0.5:
            print("  - Conclusion: Strong continuity in expert selection.")
        elif avg_jaccard > 0.2:
            print("  - Conclusion: Some continuity in expert selection.")
        else:
            print("  - Conclusion: Weak continuity in expert selection, more diverse patterns.")


    # --- 5. Quantitative Analysis: Expert Usage Frequency ---
    all_experts_list = [expert for usage in layer_expert_usage for expert in usage["experts"]]
    expert_counts = pd.Series(all_experts_list).value_counts().sort_index()

    plt.figure(figsize=(16, 8))
    expert_counts.plot(kind='bar', color='skyblue')
    plt.title(f"Total Usage Frequency per Expert for {target_layer}", fontsize=16)
    plt.xlabel("Expert Index", fontsize=12)
    plt.ylabel("Usage Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_filename_freq = f"expert_frequency_{target_layer}.png"
    plt.savefig(output_filename_freq)
    print(f"Expert usage frequency plot saved to: {output_filename_freq}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the continuity of expert usage in a MOE model.")
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the JSON file containing expert usage data."
    )
    parser.add_argument(
        "--layer", 
        type=str, 
        default="layer_0", 
        help="The target layer to analyze (e.g., 'layer_0')."
    )
    
    args = parser.parse_args()
    analyze_expert_continuity(args.file_path, args.layer) 