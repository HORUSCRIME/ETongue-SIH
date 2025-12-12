import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from typing import Dict, Any
import os

from generate_data import ETongueDataGenerator
from preprocess import prepare_data
from mlp_model import ETongueMLP, ETongueTrainer, create_data_loaders

class ETongueModelTrainer:
    def __init__(self, data_dict: Dict):
        self.data_dict = data_dict
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_mlp(self, hidden_dims=[64, 128, 64], epochs=100, lr=0.001, 
                  batch_size=32, dropout_rate=0.3, weight_decay=1e-4):
        """Train MLP model"""
        print("Training MLP model...")
        
        # Create model
        model = ETongueMLP(
            input_dim=self.data_dict['X_train'].shape[1],
            hidden_dims=hidden_dims,
            output_dim=self.data_dict['y_train'].shape[1],
            dropout_rate=dropout_rate
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            self.data_dict, batch_size=batch_size
        )
        
        # Train
        trainer = ETongueTrainer(model, self.device)
        history = trainer.fit(
            train_loader, val_loader, 
            epochs=epochs, lr=lr, weight_decay=weight_decay
        )
        
        self.models['mlp'] = {
            'model': model,
            'trainer': trainer,
            'history': history,
            'type': 'pytorch'
        }
        
        # Save model
        torch.save(model.state_dict(), 'etongue_mlp.pth')
        print("MLP model saved as 'etongue_mlp.pth'")
        
        return model, history
    
    def train_lightgbm(self, n_estimators=100, learning_rate=0.1, max_depth=6):
        """Train LightGBM model"""
        print("Training LightGBM model...")
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': learning_rate,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Train separate model for each class (multi-label)
        models = []
        for i in range(self.data_dict['y_train'].shape[1]):
            print(f"Training class {i+1}/{self.data_dict['y_train'].shape[1]}")
            
            train_data = lgb.Dataset(
                self.data_dict['X_train'], 
                label=self.data_dict['y_train'][:, i]
            )
            val_data = lgb.Dataset(
                self.data_dict['X_val'], 
                label=self.data_dict['y_val'][:, i],
                reference=train_data
            )
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=n_estimators,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            models.append(model)
        
        self.models['lightgbm'] = {
            'models': models,
            'type': 'lightgbm'
        }
        
        # Save models
        joblib.dump(models, 'etongue_lightgbm.pkl')
        print("LightGBM models saved as 'etongue_lightgbm.pkl'")
        
        return models
    
    def train_random_forest(self, n_estimators=100, max_depth=10, min_samples_split=5):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        # Use MultiOutputClassifier for multi-label
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        
        model = MultiOutputClassifier(rf)
        model.fit(self.data_dict['X_train'], self.data_dict['y_train'])
        
        self.models['random_forest'] = {
            'model': model,
            'type': 'sklearn'
        }
        
        # Save model
        joblib.dump(model, 'etongue_rf.pkl')
        print("Random Forest model saved as 'etongue_rf.pkl'")
        
        return model
    
    def train_xgboost(self, n_estimators=100, learning_rate=0.1, max_depth=6):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # Train separate model for each class
        models = []
        for i in range(self.data_dict['y_train'].shape[1]):
            print(f"Training class {i+1}/{self.data_dict['y_train'].shape[1]}")
            
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
                eval_metric='logloss'
            )
            
            model.fit(
                self.data_dict['X_train'], 
                self.data_dict['y_train'][:, i]
            )
            models.append(model)
        
        self.models['xgboost'] = {
            'models': models,
            'type': 'xgboost'
        }
        
        # Save models
        joblib.dump(models, 'etongue_xgb.pkl')
        print("XGBoost models saved as 'etongue_xgb.pkl'")
        
        return models
    
    def train_all_models(self):
        """Train all model types"""
        print("Training all models...")
        
        # Train MLP
        self.train_mlp(epochs=50, lr=0.001, batch_size=32)
        
        # Train LightGBM
        self.train_lightgbm(n_estimators=100, learning_rate=0.1)
        
        # Train Random Forest
        self.train_random_forest(n_estimators=100, max_depth=10)
        
        # Train XGBoost
        self.train_xgboost(n_estimators=100, learning_rate=0.1)
        
        print("All models trained successfully!")
        
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history for MLP"""
        if 'mlp' not in self.models:
            print("MLP model not trained yet")
            return
        
        history = self.models['mlp']['history']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # F1 score
        ax2.plot(history['val_f1'], label='Validation F1', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Validation F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history saved as '{save_path}'")

def main():
    """Main training pipeline"""
    print("E-Tongue ML System Training Pipeline")
    print("=" * 50)
    
    # Generate dataset if not exists
    if not os.path.exists('etongue_dataset.csv'):
        print("Generating synthetic dataset...")
        generator = ETongueDataGenerator(random_state=42)
        df = generator.generate_dataset(
            n_samples_per_class=1200,
            n_mixture_samples=800
        )
        df.to_csv('etongue_dataset.csv', index=False)
        print("Dataset generated and saved")
    
    # Prepare data
    print("Preparing data...")
    data_dict = prepare_data('etongue_dataset.csv', test_size=0.2, val_size=0.1)
    
    # Initialize trainer
    trainer = ETongueModelTrainer(data_dict)
    
    # Train all models
    trainer.train_all_models()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save preprocessing info
    joblib.dump(data_dict['preprocessor'], 'etongue_preprocessor.pkl')
    
    # Save class names and feature names
    metadata = {
        'class_names': data_dict['class_names'],
        'feature_names': data_dict['feature_names'],
        'input_dim': len(data_dict['feature_names']),
        'output_dim': len(data_dict['class_names'])
    }
    joblib.dump(metadata, 'etongue_metadata.pkl')
    
    print("\nTraining completed!")
    print("Saved files:")
    print("- etongue_mlp.pth (PyTorch MLP model)")
    print("- etongue_lightgbm.pkl (LightGBM models)")
    print("- etongue_rf.pkl (Random Forest model)")
    print("- etongue_xgb.pkl (XGBoost models)")
    print("- etongue_preprocessor.pkl (Data preprocessor)")
    print("- etongue_metadata.pkl (Model metadata)")
    
    return trainer, data_dict

if __name__ == "__main__":
    trainer, data_dict = main()