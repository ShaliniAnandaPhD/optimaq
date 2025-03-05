##This script serves as the main entry point for the OptimaQ addiction prediction system,
##handling training, prediction, and evaluation workflows.

import os
import argparse
import logging
import json
import pandas as pd
from datetime import datetime

# Import OptimaQ modules
from optimaq_model import QPredictionModel
from optimaq_data import DataProcessor
from optimaq_evaluation import ModelEvaluator
from optimaq_visualization import ResultVisualizer
from optimaq_utils import setup_logging, save_model, load_model

def parse_arguments():
    ##Parse command line arguments.
    parser = argparse.ArgumentParser(description='OptimaQ: Addiction Prediction System')
    parser.add_argument('--data_path', type=str, required=True, help='Path to patient data CSV file')
    parser.add_argument('--config', type=str, default='config/default_config.json', help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate'], default='train', help='Mode of operation')
    parser.add_argument('--model_path', type=str, help='Path to pre-trained model')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')
    return parser.parse_args()

def main():
    ##Main execution function for OptimaQ system.
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, 'optimaq.log'), args.log_level)
    logging.info("Starting OptimaQ system...")
    
    # Load configuration
    with open(args.config) as config_file:
        config = json.load(config_file)
    logging.info(f"Loaded configuration from {args.config}")
    
    # Initialize data processor
    data_processor = DataProcessor(args.data_path, config['preprocessing'])
    data_processor.load_data()
    data_processor.preprocess()
    
    if args.mode == 'train':
        model = QPredictionModel(**config['model'])
        model.train(*data_processor.prepare_train_test_split(), config['training']['num_episodes'])
        save_model(model, os.path.join(args.output_dir, 'optimaq_model.pkl'))
    
    elif args.mode == 'predict':
        model = load_model(args.model_path)
        predictions = model.predict_batch(data_processor.get_processed_features())
        pd.DataFrame({'patient_id': data_processor.get_patient_ids(), 'prediction': predictions}).to_csv(
            os.path.join(args.output_dir, 'predictions.csv'), index=False)
    
    elif args.mode == 'evaluate':
        model = load_model(args.model_path)
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(*data_processor.get_features_and_labels())
        with open(os.path.join(args.output_dir, 'evaluation.json'), 'w') as eval_file:
            json.dump(metrics, eval_file, indent=4)
    
    logging.info("OptimaQ execution completed.")

if __name__ == "__main__":
    main()
