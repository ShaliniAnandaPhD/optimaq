# OptimaQ: AI-Powered Addiction Prediction System

## Overview
Hey there! Welcome to OptimaQ - our advanced system that uses reinforcement learning (specifically Q-learning) to predict opioid addiction patterns and suggest treatment plans. We've built this to categorize patients into four main groups:
- Healthy
- Continuous Use
- Relapsing
- Recovering

The cool thing about OptimaQ is that it models addiction decision-making by analyzing patient prescriptions, claims data, and medical history. This helps us predict potential relapse or recovery trajectories, which is pretty powerful stuff.

## Key Features
- **Smart Learning Model**: We've implemented a dual Q-learning approach that tracks both positive and negative rewards for addiction-related decisions.  
- **Data Processing**: We handle the messy work of cleaning and transforming medical claims and prescription records.  
- **Prediction Power**: The system forecasts addiction risk using state-action modeling.  
- **Treatment Planning**: We can generate optimal treatment paths based on what the model learns.  
- **Solid Evaluation**: We measure everything with accuracy, sensitivity, specificity, and clinical risk metrics.  
- **Visual Results**: You get confusion matrices, ROC curves, and reward function graphs to understand what's happening.  

## Project Structure
```
optimaq/
│── optimaq_main.py           # Main execution script
│── optimaq_model.py          # Core Q-learning addiction prediction model
│── optimaq_data.py           # Data processing and feature extraction
│── optimaq_evaluation.py     # Model evaluation metrics and validation
│── optimaq_visualization.py  # Graphical analysis and reports
│── optimaq_utils.py          # Helper utilities (logging, saving/loading models)
│── config/
│   ├── default_config.json   # Model and training configurations
│── results/                  # Output predictions and evaluations
│── README.md                 # Project documentation
```

## Installation
### Prerequisites
You'll need Python 3.8 or newer. Just install the dependencies with:
```sh
pip install -r requirements.txt
```

### Dependencies
Nothing fancy here, just the usual suspects:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm
- joblib
- argparse
- json

## Usage
### Training the Model
To train OptimaQ on your patient data, run:
```sh
python optimaq_main.py --data_path data/patient_data.csv --mode train --output_dir results/
```

This will:
- Load and process your patient data
- Train the Q-learning model on addiction categories
- Save the trained model in your results folder

### Making Predictions
Want to predict addiction risk for new patients? Just run:
```sh
python optimaq_main.py --data_path data/new_patients.csv --mode predict --model_path results/optimaq_model.pkl
```

This will:
- Load your trained model
- Predict addiction categories for new patients
- Save results to results/predictions.csv

### Evaluating Performance
To see how well OptimaQ is doing:
```sh
python optimaq_main.py --data_path data/test_data.csv --mode evaluate --model_path results/optimaq_model.pkl
```

This generates accuracy scores, precision, recall, F1-score, plus a confusion matrix and category analysis, all saved to results/evaluation.json.

## Modules

### optimaq_model.py
This is where the magic happens. We implement Q-learning for addiction prediction, maintain dual Q-tables for tracking positive and negative rewards, use an epsilon-greedy strategy for action selection, and support batch predictions and treatment simulations.

### optimaq_data.py
Data wrangling central! This module loads patient claims and prescription data, cleans everything up, normalizes values, and engineers features like prescription intensity and opioid usage patterns. It also handles encoding those pesky ICD diagnosis codes.

### optimaq_evaluation.py
The truth-teller module. It calculates accuracy, precision, recall, and F1-score, implements addiction-specific metrics (like false negatives in high-risk groups), and supports cross-validation and sensitivity analysis.

### optimaq_visualization.py
Making things pretty! Generates confusion matrices, ROC curves, reward plots, and analyzes patient state transitions and Q-value distributions.

### optimaq_utils.py
The helper module that handles logging, saving/loading models, argument parsing, and provides functions for model persistence and debugging.

## Example Workflow

### Step 1: Train Model
```sh
python optimaq_main.py --data_path data/patient_data.csv --mode train
```

### Step 2: Predict Addiction Categories
```sh
python optimaq_main.py --data_path data/new_patients.csv --mode predict --model_path results/optimaq_model.pkl
```

### Step 3: Evaluate Model Performance
```sh
python optimaq_main.py --data_path data/test_data.csv --mode evaluate --model_path results/optimaq_model.pkl
```

### Step 4: Visualize Results
```sh
python optimaq_visualization.py --evaluation_file results/evaluation.json
```

## Results & Expected Outcomes
When everything's running smoothly, you'll get:
- Patient categorization across our four states (healthy, continuous use, relapsing, recovering)
- Treatment simulations showing how interventions might impact patient states
- Clinical insights to help doctors adjust prescriptions and monitor relapse risks

## License
MIT License - Feel free to use, modify, and share!
