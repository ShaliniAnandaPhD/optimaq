## OptimaQ Model Evaluation Module

## This module provides tools for evaluating model performance, conducting
## sensitivity analyses, and comparing different model configurations.


import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        logging.info("Initialized ModelEvaluator")

    def evaluate(self, X, y):
        logging.info("Evaluating model performance")
        y_pred = self.model.predict_batch(X)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        cm = confusion_matrix(y, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }

        logging.info(f"Evaluation complete: Accuracy={accuracy:.4f}, F1-Score={f1:.4f}")
        return metrics
