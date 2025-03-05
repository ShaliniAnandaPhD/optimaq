## Utility Module

## This module provides utility functions for logging, saving/loading models,
## and managing configurations.


import logging
import pickle
import os

def setup_logging(log_path, log_level="INFO"):
    logging.basicConfig(
        filename=log_path,
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("Logging setup complete.")

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {file_path}")

def load_model(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    logging.info(f"Model loaded from {file_path}")
    return model
