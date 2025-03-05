## Visualization Module

## This module provides functions to visualize learning curves,
## confusion matrices, and feature importance for the model.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

class ResultVisualizer:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir

    def plot_learning_curves(self, learning_curves):
        plt.figure(figsize=(10, 5))
        plt.plot(learning_curves['rewards'], label="Average Reward")
        plt.plot(learning_curves['accuracies'], label="Accuracy")
        plt.xlabel("Episodes")
        plt.ylabel("Metrics")
        plt.legend()
        plt.title("Learning Curves")
        plt.savefig(f"{self.output_dir}/learning_curves.png")
        plt.close()

    def plot_confusion_matrix(self, cm, labels):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(f"{self.output_dir}/confusion_matrix.png")
        plt.close()

    def plot_feature_importance(self, feature_importances, feature_names):
        indices = np.argsort(feature_importances)[::-1]
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(feature_importances)), feature_importances[indices], align="center")
        plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], rotation=90)
        plt.xlabel("Feature Importance")
        plt.ylabel("Value")
        plt.title("Feature Importance Analysis")
        plt.savefig(f"{self.output_dir}/feature_importance.png")
        plt.close()
