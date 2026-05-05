import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import re

# history dla 3 modeli

history1 = pd.read_csv("results/history_CustomCNN_80_20.csv")
history2 = pd.read_csv("results/history_MobileNetV2_80_20.csv")
history3 = pd.read_csv("results/history_VGG16_80_20.csv")

histories = {
    "CustomCNN": history1,
    "MobileNetV2": history2,
    "VGG16": history3
}

import matplotlib.pyplot as plt

for name, hist in histories.items():
    
    # 🔸 Accuracy
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(hist["epoch"], hist["accuracy"], '--', label="train")
    ax.plot(hist["epoch"], hist["val_accuracy"], label="val")
    
    ax.set_title(f"{name} - Accuracy vs Epochs (80/20)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(f"acc_vs_ep_{name}.png", dpi=300)
    plt.close(fig)

    # 🔸 Loss
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(hist["epoch"], hist["loss"], '--', label="train")
    ax.plot(hist["epoch"], hist["val_loss"], label="val")
    
    ax.set_title(f"{name} - Loss vs Epochs (80/20)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(f"loss_vs_ep_{name}.png", dpi=300)
    plt.close(fig)

cm1 = pd.read_csv("results/confusion_matrix_CustomCNN_80_20.csv", index_col=0)
cm2 = pd.read_csv("results/confusion_matrix_MobileNetV2_80_20.csv", index_col=0)
cm3 = pd.read_csv("results/confusion_matrix_VGG16_80_20.csv", index_col=0)


cms = {
    "CustomCNN": cm1.values,
    "MobileNetV2": cm2.values,  # podmień
    "VGG16": cm3.values   # podmień
}

labels = cm1.columns.tolist()
for name, cm in cms.items():

    cm = cm.astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_title(f"{name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    fig.tight_layout()
    fig.savefig(f"conf_matrix_{name}.png", dpi=300)
    plt.close(fig)

def load_accuracy_vs_split(model_name):
    files = glob.glob(f"results/history_{model_name}_*.csv")
    
    splits = []
    val_accuracies = []

    for file in files:
        # wyciągnięcie np. "80" z "80_20"
        match = re.search(r"_(\d+)_\d+\.csv", file)
        if match:
            train_split = int(match.group(1))
        else:
            continue

        df = pd.read_csv(file)

        final_val_acc = df["val_accuracy"].iloc[-1]

        splits.append(train_split)
        val_accuracies.append(final_val_acc)

    # sortowanie po splitach
    splits, val_accuracies = zip(*sorted(zip(splits, val_accuracies)))

    return splits, val_accuracies

splits_A, acc_A = load_accuracy_vs_split("CustomCNN")
splits_B, acc_B = load_accuracy_vs_split("MobileNetV2")
splits_C, acc_C = load_accuracy_vs_split("VGG16")

plt.figure(figsize=(8,5))

plt.plot(splits_A, acc_A, marker='o', label="CustomCNN")
plt.plot(splits_B, acc_B, marker='o', label="MobileNetV2")
plt.plot(splits_C, acc_C, marker='o', label="VGG16")

plt.xlabel("Training Data (%)")
plt.ylabel("Final Validation Accuracy")
plt.title("Accuracy vs Training Set Size")
plt.legend()
plt.grid(True)

plt.savefig("acc_vs_split.png")