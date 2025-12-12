import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class ETonguePreprocessor:
    def __init__(self, scaler_type='standard', outlier_threshold=3.0):
        self.scaler_type = scaler_type
        self.outlier_threshold = outlier_threshold
        self.scaler = None
        self.feature_columns = None
        self.taste_classes = ['sweet', 'salty', 'sour', 'bitter', 'umami', 'astringent', 'bland']
        
    def _init_scaler(self):
        """Initialize the appropriate scaler"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def detect_outliers(self, X: np.ndarray) -> np.ndarray:
        """Detect outliers using Z-score method"""
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        outliers = np.any(z_scores > self.outlier_threshold, axis=1)
        return outliers
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in sensor data"""
        sensor_cols = [col for col in df.columns if col.startswith('ch_')]
        
        # Fill missing values with column median
        for col in sensor_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
                
        return df
    
    def apply_drift_correction(self, X: np.ndarray, window_size=50) -> np.ndarray:
        """Apply rolling mean normalization for drift correction"""
        X_corrected = X.copy()
        
        for i in range(X.shape[1]):  # for each sensor channel
            rolling_mean = pd.Series(X[:, i]).rolling(
                window=window_size, min_periods=1, center=True
            ).mean()
            X_corrected[:, i] = X[:, i] - rolling_mean + np.mean(X[:, i])
            
        return X_corrected
    
    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        """Convert multi-label strings to binary matrix"""
        n_samples = len(labels)
        n_classes = len(self.taste_classes)
        y_encoded = np.zeros((n_samples, n_classes))
        
        for i, label_str in enumerate(labels):
            if pd.isna(label_str) or label_str == 'unknown':
                continue  # skip unknown labels
                
            tastes = label_str.split(';')
            for taste in tastes:
                taste = taste.strip()
                if taste in self.taste_classes:
                    class_idx = self.taste_classes.index(taste)
                    y_encoded[i, class_idx] = 1
                    
        return y_encoded
    
    def decode_labels(self, y_encoded: np.ndarray, threshold=0.5) -> List[str]:
        """Convert binary matrix back to multi-label strings"""
        labels = []
        for row in y_encoded:
            active_tastes = []
            for i, prob in enumerate(row):
                if prob > threshold:
                    active_tastes.append(self.taste_classes[i])
            
            if active_tastes:
                labels.append(';'.join(active_tastes))
            else:
                labels.append('bland')  # default if no taste detected
                
        return labels
    
    def fit_transform(self, df: pd.DataFrame, apply_drift_correction=True) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform data"""
        # Handle missing values
        df = self.handle_missing_values(df.copy())
        
        # Extract sensor features
        self.feature_columns = [col for col in df.columns if col.startswith('ch_')]
        X = df[self.feature_columns].values.astype(np.float32)
        
        # Apply drift correction if requested
        if apply_drift_correction:
            X = self.apply_drift_correction(X)
        
        # Remove outliers
        outlier_mask = self.detect_outliers(X)
        print(f"Detected {np.sum(outlier_mask)} outliers ({np.mean(outlier_mask)*100:.1f}%)")
        
        X_clean = X[~outlier_mask]
        y_clean = df.loc[~outlier_mask, 'label']
        
        # Fit and apply scaler
        self._init_scaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Encode labels
        y_encoded = self.encode_labels(y_clean)
        
        print(f"Preprocessed data shape: {X_scaled.shape}")
        print(f"Label distribution: {np.sum(y_encoded, axis=0)}")
        
        return X_scaled, y_encoded
    
    def transform(self, df: pd.DataFrame, apply_drift_correction=True) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Handle missing values
        df = self.handle_missing_values(df.copy())
        
        # Extract features
        X = df[self.feature_columns].values.astype(np.float32)
        
        # Apply drift correction if requested
        if apply_drift_correction:
            X = self.apply_drift_correction(X)
        
        # Apply scaling
        X_scaled = self.scaler.transform(X)
        
        return X_scaled

def prepare_data(csv_path: str, test_size=0.2, val_size=0.1, 
                scaler_type='standard', random_state=42) -> Dict:
    """Complete data preparation pipeline"""
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    
    # Initialize preprocessor
    preprocessor = ETonguePreprocessor(scaler_type=scaler_type)
    
    # Fit and transform
    X, y = preprocessor.fit_transform(df)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y.argmax(axis=1)
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
        stratify=y_temp.argmax(axis=1)
    )
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.feature_columns,
        'class_names': preprocessor.taste_classes
    }

def main():
    """Test preprocessing pipeline"""
    # Generate sample data first
    from generate_data import ETongueDataGenerator
    
    generator = ETongueDataGenerator()
    df = generator.generate_dataset(n_samples_per_class=200, n_mixture_samples=100)
    df.to_csv('test_data.csv', index=False)
    
    # Test preprocessing
    data_dict = prepare_data('test_data.csv')
    
    print("Preprocessing completed successfully!")
    print(f"Feature shape: {data_dict['X_train'].shape}")
    print(f"Label shape: {data_dict['y_train'].shape}")
    
    return data_dict

if __name__ == "__main__":
    data = main()