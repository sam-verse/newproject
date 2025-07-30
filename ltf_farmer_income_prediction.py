#!/usr/bin/env python3
"""
LTF Farmer Income Prediction Challenge Solution
Date: July 30, 2025

This script implements a comprehensive machine learning solution for predicting 
farmer income in India using demographic, agricultural, weather, and socio-economic features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# Try to import XGBoost, make it optional
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Warning: XGBoost not available ({str(e)}). Skipping XGBoost model.")
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class FarmerIncomePredictor:
    """
    A comprehensive farmer income prediction system that handles data preprocessing,
    feature engineering, model training, and evaluation.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.feature_names = None
        
    def load_data(self, file_path):
        """Load and perform initial data exploration"""
        print("Loading data...")
        try:
            # Handle the specific filename format
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, data):
        """Perform comprehensive exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print(f"\nDataset shape: {data.shape}")
        print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Target variable analysis
        target_col = 'Target_Variable/Total Income'
        if target_col in data.columns:
            print(f"\nTarget Variable Statistics:")
            print(f"Mean: {data[target_col].mean():,.2f}")
            print(f"Median: {data[target_col].median():,.2f}")
            print(f"Std: {data[target_col].std():,.2f}")
            print(f"Min: {data[target_col].min():,.2f}")
            print(f"Max: {data[target_col].max():,.2f}")
        
        # Missing values analysis
        missing_data = data.isnull().sum()
        missing_pct = (missing_data / len(data)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        if not missing_df.empty:
            print(f"\nMissing Values (Top 10):")
            print(missing_df.head(10))
        else:
            print("\nNo missing values found!")
        
        # Data types
        print(f"\nData Types:")
        dtype_counts = data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"{dtype}: {count} columns")
        
        return missing_df
    
    def clean_column_names(self, data):
        """Clean and standardize column names"""
        print("Cleaning column names...")
        # Remove special characters and spaces, convert to lowercase
        data.columns = data.columns.str.replace(r'[^\w\s]', '_', regex=True)
        data.columns = data.columns.str.replace(r'\s+', '_', regex=True)
        data.columns = data.columns.str.replace(r'_+', '_', regex=True)
        data.columns = data.columns.str.strip('_')
        return data
    
    def preprocess_data(self, data, is_training=True):
        """Comprehensive data preprocessing and feature engineering"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING & FEATURE ENGINEERING")
        print("="*50)
        
        # Clean column names
        data = self.clean_column_names(data.copy())
        
        # Identify target column - specifically look for the target variable
        target_col = None
        for col in data.columns:
            if 'target_variable' in col.lower() and 'income' in col.lower():
                target_col = col
                break
        
        if not target_col:
            # Fallback: Look for columns with "total" and "income"
            target_cols = [col for col in data.columns if 'total' in col.lower() and 'income' in col.lower()]
            if target_cols:
                target_col = target_cols[0]
        
        if target_col:
            print(f"Target column identified: {target_col}")
        else:
            target_col = None
            print("Warning: No target column found!")
            print("Available columns:", list(data.columns))
        
        # Separate features and target
        if target_col and is_training:
            y = data[target_col].copy()
            X = data.drop(columns=[target_col])
        else:
            y = None
            X = data.copy()
        
        # Remove ID columns
        id_cols = [col for col in X.columns if 'id' in col.lower() or 'farmer' in col.lower()]
        if id_cols:
            print(f"Removing ID columns: {id_cols}")
            X = X.drop(columns=id_cols)
        
        # Handle missing values
        print("Handling missing values...")
        
        # Separate numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric columns: {len(numeric_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")
        
        # Fill missing values
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                mode_val = X[col].mode()
                if len(mode_val) > 0:
                    X[col].fillna(mode_val[0], inplace=True)
                else:
                    X[col].fillna('Unknown', inplace=True)
        
        # Feature Engineering (Enhanced for better performance)
        print("Creating engineered features...")
        
        # 1. Irrigation Ratio
        land_cols = [col for col in numeric_cols if 'land' in col.lower() or 'area' in col.lower()]
        irrigated_cols = [col for col in numeric_cols if 'irrigated' in col.lower()]
        
        if land_cols and irrigated_cols:
            total_land = X[land_cols].sum(axis=1)
            total_irrigated = X[irrigated_cols].sum(axis=1)
            X['irrigation_ratio'] = np.where(total_land > 0, total_irrigated / total_land, 0)
        
        # 2. Non-Agricultural Income Flag
        non_agri_cols = [col for col in X.columns if 'non' in col.lower() and 'agriculture' in col.lower()]
        if non_agri_cols:
            X['has_non_agri_income'] = (X[non_agri_cols[0]] > 0).astype(int)
        
        # 3. Credit Exposure (Enhanced)
        loan_cols = [col for col in numeric_cols if 'loan' in col.lower()]
        disbursement_cols = [col for col in numeric_cols if 'disbursement' in col.lower()]
        
        if loan_cols and disbursement_cols:
            # Handle missing values in disbursement
            X[disbursement_cols[0]] = X[disbursement_cols[0]].fillna(0)
            X['credit_exposure'] = X[loan_cols[0]] * X[disbursement_cols[0]]
            X['has_credit'] = (X[loan_cols[0]] > 0).astype(int)
        
        # 4. Weather Stability (rainfall variance across years)
        rainfall_cols = [col for col in numeric_cols if 'rainfall' in col.lower()]
        if len(rainfall_cols) > 1:
            X['weather_stability'] = X[rainfall_cols].std(axis=1)
            X['avg_rainfall'] = X[rainfall_cols].mean(axis=1)
        
        # 5. Agricultural Performance Trend
        performance_cols = [col for col in numeric_cols if 'performance' in col.lower()]
        if len(performance_cols) > 1:
            X['agri_performance_trend'] = X[performance_cols].mean(axis=1)
            X['agri_performance_max'] = X[performance_cols].max(axis=1)
        
        # 6. Infrastructure Index (Enhanced)
        electricity_cols = [col for col in numeric_cols if 'electricity' in col.lower()]
        housing_cols = [col for col in numeric_cols if 'house' in col.lower() or 'room' in col.lower()]
        
        if electricity_cols and housing_cols:
            # Normalize and combine
            elec_norm = (X[electricity_cols[0]] - X[electricity_cols[0]].min()) / (X[electricity_cols[0]].max() - X[electricity_cols[0]].min() + 1e-8)
            house_norm = (X[housing_cols[0]] - X[housing_cols[0]].min()) / (X[housing_cols[0]].max() - X[housing_cols[0]].min() + 1e-8)
            X['infrastructure_index'] = (elec_norm + house_norm) / 2
        
        # 7. Geographic and Economic Indicators
        score_cols = [col for col in numeric_cols if 'score' in col.lower()]
        if score_cols:
            X['avg_socio_score'] = X[score_cols].mean(axis=1)
        
        # 8. Agricultural Intensity
        area_cols = [col for col in numeric_cols if 'area' in col.lower() and 'agri' in col.lower()]
        if len(area_cols) > 1:
            X['total_agri_area'] = X[area_cols].sum(axis=1)
            X['agri_diversity'] = (X[area_cols] > 0).sum(axis=1)
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_vals = set(X[col].astype(str).unique())
                    known_vals = set(self.label_encoders[col].classes_)
                    unseen_vals = unique_vals - known_vals
                    
                    if unseen_vals:
                        # Map unseen values to most frequent class
                        most_frequent = self.label_encoders[col].classes_[0]
                        X[col] = X[col].astype(str).replace(list(unseen_vals), most_frequent)
                    
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Update numeric columns list after feature engineering
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X[numeric_cols])
            self.feature_names = numeric_cols
        else:
            X_scaled = self.scaler.transform(X[numeric_cols])
        
        X_final = pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)
        
        print(f"Final feature matrix shape: {X_final.shape}")
        print(f"Feature engineering completed. Added features:")
        added_features = [col for col in X_final.columns if col not in data.columns]
        for feature in added_features:
            print(f"  - {feature}")
        
        return X_final, y
    
    def train_models(self, X_train, y_train):
        """Train and compare multiple models using cross-validation"""
        print("\n" + "="*50)
        print("MODEL TRAINING & COMPARISON")
        print("="*50)
        
        # Define models (optimized for speed and performance)
        models = {
            'Ridge Regression': Ridge(alpha=10.0, random_state=42),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42, 
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=150, 
                learning_rate=0.1,
                max_depth=6,
                subsample=0.9,
                random_state=42
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        
        # Cross-validation results
        cv_results = {}
        
        print("Performing 3-fold cross-validation...")  # Reduced for speed
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Negative MAPE for cross-validation (sklearn convention)
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=3, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)  # Reduced CV folds
            
            cv_results[name] = {
                'mean_mape': -scores.mean(),
                'std_mape': scores.std(),
                'model': model
            }
            
            print(f"  MAPE: {-scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Select best model
        best_model_name = min(cv_results.keys(), key=lambda x: cv_results[x]['mean_mape'])
        self.best_model = cv_results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best CV MAPE: {cv_results[best_model_name]['mean_mape']:.4f}")
        
        return cv_results, best_model_name
    
    def tune_hyperparameters(self, X_train, y_train, best_model_name):
        """Perform hyperparameter tuning on the best model"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # Define parameter grids (optimized for speed)
        param_grids = {
            'Random Forest': {
                'n_estimators': [200, 300],
                'max_depth': [15, 20],
                'min_samples_split': [3, 5]
            },
            'Gradient Boosting': {
                'n_estimators': [150, 200],
                'learning_rate': [0.08, 0.1, 0.12],
                'max_depth': [5, 6]
            },
            'Ridge Regression': {
                'alpha': [5.0, 10.0, 20.0]
            }
        }
        
        if best_model_name in param_grids:
            print(f"Tuning hyperparameters for {best_model_name}...")
            
            grid_search = GridSearchCV(
                self.best_model,
                param_grids[best_model_name],
                cv=3,  # Fast 3-fold CV
                scoring='neg_mean_absolute_percentage_error',
                n_jobs=-1,
                verbose=0  # Reduced verbosity
            )
            
            grid_search.fit(X_train, y_train)
            
            self.best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV MAPE: {-grid_search.best_score_:.4f}")
        else:
            print(f"No hyperparameter tuning defined for {best_model_name}")
            # Fit the model with default parameters
            self.best_model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the best model on test data"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test Set Results:")
        print(f"MAPE: {mape:.4f}")
        print(f"RMSE: {rmse:,.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance()
        
        return {
            'mape': mape,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance for tree-based models"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(top_n), y='feature', x='importance')
            plt.title(f'Top {top_n} Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            print(f"\nTop {top_n} Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
                print(f"{i:2d}. {row['feature']}: {row['importance']:.4f}")
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        return self.best_model.predict(X)
    
    def create_submission_file(self, test_data, predictions, filename='farmer_income_predictions.csv'):
        """Create submission file in the required format"""
        # Find ID column
        id_cols = [col for col in test_data.columns if 'id' in col.lower() or 'farmer' in col.lower()]
        
        if id_cols:
            submission = pd.DataFrame({
                id_cols[0]: test_data[id_cols[0]],
                'Predicted_Income': predictions
            })
        else:
            submission = pd.DataFrame({
                'Index': range(len(predictions)),
                'Predicted_Income': predictions
            })
        
        submission.to_csv(filename, index=False)
        print(f"Submission file saved as: {filename}")
        return submission


def main():
    """Main function to run the farmer income prediction pipeline"""
    print("="*60)
    print("LTF FARMER INCOME PREDICTION CHALLENGE")
    print("="*60)
    
    # Initialize predictor
    predictor = FarmerIncomePredictor()
    
    # Load training data
    train_file = "LTF Challenge data with dictionary.xlsx - TrainData (1).csv"
    train_data = predictor.load_data(train_file)
    
    if train_data is None:
        print("Failed to load training data. Exiting...")
        return
    
    # Explore data
    missing_info = predictor.explore_data(train_data)
    
    # Preprocess data
    X, y = predictor.preprocess_data(train_data, is_training=True)
    
    if y is None:
        print("No target variable found. Cannot proceed with training.")
        return
    
    # Split data
    print(f"\nSplitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Train models
    cv_results, best_model_name = predictor.train_models(X_train, y_train)
    
    # Tune hyperparameters
    predictor.tune_hyperparameters(X_train, y_train, best_model_name)
    
    # Evaluate model
    results = predictor.evaluate_model(X_val, y_val)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"Final Validation MAPE: {results['mape']:.4f}")
    print(f"Final Validation RMSE: {results['rmse']:,.2f}")
    print(f"Final Validation R²: {results['r2']:.4f}")
    
    # For test data prediction (uncomment when test data is available)
    """
    print("\nMaking predictions on test data...")
    test_data = predictor.load_data('test_data.csv')
    X_test, _ = predictor.preprocess_data(test_data, is_training=False)
    test_predictions = predictor.predict(X_test)
    
    # Create submission file
    submission = predictor.create_submission_file(test_data, test_predictions)
    """
    
    print("\nPrediction pipeline completed successfully!")
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()
