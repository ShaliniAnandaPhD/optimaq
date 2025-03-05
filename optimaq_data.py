
## Data Processing Module

## This module handles data loading, preprocessing, and feature engineering
## for the OptimaQ addiction prediction system.


import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataProcessor:
    def __init__(self, data_path, config=None):
        self.data_path = data_path
        self.config = config or {}
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = None
        self.label_column = None
        self.patient_ids = None

        logging.info(f"Initialized DataProcessor with data path: {data_path}")

    def load_data(self):
        logging.info(f"Loading data from {self.data_path}")
        try:
            self.raw_data = pd.read_csv(self.data_path) if self.data_path.endswith('.csv') else None
            logging.info(f"Loaded data with {self.raw_data.shape[0]} rows and {self.raw_data.shape[1]} columns")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess(self):
        logging.info("Starting data preprocessing")
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Handle missing values
        self.raw_data.fillna(self.raw_data.median(), inplace=True)

        # Extract features and labels
        self.label_column = self.config.get('label_column', 'label')
        self.feature_columns = [col for col in self.raw_data.columns if col != self.label_column]

        X = self.raw_data[self.feature_columns]
        y = self.raw_data[self.label_column] if self.label_column in self.raw_data.columns else None

        # Preprocessing pipeline
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        self.processed_data = {
            'X': self.preprocessor.fit_transform(X),
            'y': y
        }

        logging.info(f"Preprocessing complete. Processed data shape: {self.processed_data['X'].shape}")
        return self.processed_data['X'], y

    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        X, y = self.processed_data['X'], self.processed_data['y']
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
