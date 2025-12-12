import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import random

class ETongueDataGenerator:
    def __init__(self, n_channels=18, random_state=42):
        self.n_channels = n_channels
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define taste-specific sensor patterns
        self.taste_patterns = {
            'sweet': {'primary_channels': [2, 4, 8], 'intensity': 0.8, 'noise': 0.1},
            'salty': {'primary_channels': [1, 6], 'intensity': 0.9, 'noise': 0.15},
            'sour': {'primary_channels': [11, 12], 'intensity': 0.85, 'noise': 0.12},
            'bitter': {'primary_channels': [14], 'intensity': 0.7, 'noise': 0.25},
            'umami': {'primary_channels': [0, 3], 'intensity': 0.75, 'noise': 0.1},
            'astringent': {'primary_channels': [15, 16], 'intensity': 0.6, 'noise': 0.2},
            'bland': {'primary_channels': [], 'intensity': 0.1, 'noise': 0.05}
        }
        
    def generate_base_vector(self, taste: str) -> np.ndarray:
        """Generate base sensor response for a single taste"""
        vector = np.random.normal(0.1, 0.05, self.n_channels)  # baseline
        pattern = self.taste_patterns[taste]
        
        for ch in pattern['primary_channels']:
            vector[ch] = np.random.normal(pattern['intensity'], pattern['noise'])
            
        # Add cross-channel correlations
        if taste == 'sweet':
            vector[9] = vector[2] * 0.6 + np.random.normal(0, 0.1)
        elif taste == 'sour':
            vector[13] = vector[11] * 0.7 + np.random.normal(0, 0.1)
            
        return np.clip(vector, 0, 1)
    
    def add_sensor_effects(self, data: np.ndarray, drift_level=0.02, 
                          noise_level=0.05, dropout_prob=0.01) -> np.ndarray:
        """Add realistic sensor effects"""
        n_samples, n_channels = data.shape
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, data.shape)
        data += noise
        
        # Add multiplicative scaling
        scaling = np.random.normal(1.0, 0.05, (n_samples, n_channels))
        data *= scaling
        
        # Add drift (slow trend)
        drift = np.cumsum(np.random.normal(0, drift_level/n_samples, 
                                         (n_samples, n_channels)), axis=0)
        data += drift
        
        # Random dropouts
        dropout_mask = np.random.random(data.shape) < dropout_prob
        data[dropout_mask] = 0
        
        return np.clip(data, 0, 1)
    
    def generate_mixture(self, tastes: List[str], weights: List[float] = None) -> np.ndarray:
        """Generate multi-label sample by mixing taste vectors"""
        if weights is None:
            weights = [1.0] * len(tastes)
        weights = np.array(weights) / np.sum(weights)
        
        mixture = np.zeros(self.n_channels)
        for taste, weight in zip(tastes, weights):
            mixture += weight * self.generate_base_vector(taste)
            
        # Add interaction noise
        interaction_noise = np.random.normal(0, 0.05, self.n_channels)
        mixture += interaction_noise
        
        return np.clip(mixture, 0, 1)
    
    def generate_dataset(self, n_samples_per_class=1000, n_mixture_samples=500,
                        class_distribution=None, noise_level=0.05, 
                        drift_level=0.02) -> pd.DataFrame:
        """Generate complete synthetic dataset"""
        
        tastes = list(self.taste_patterns.keys())
        data = []
        labels = []
        metadata = []
        
        # Generate pure class samples
        for taste in tastes:
            if taste == 'bland':
                n_samples = n_samples_per_class // 2  # fewer bland samples
            else:
                n_samples = n_samples_per_class
                
            for i in range(n_samples):
                vector = self.generate_base_vector(taste)
                data.append(vector)
                labels.append(taste)
                metadata.append({
                    'batch_id': f'B{i//100:03d}',
                    'sample_id': f'{taste}_{i:04d}',
                    'temp_c': np.random.normal(25, 2),
                    'humidity_pct': np.random.normal(45, 5),
                    'instrument_id': f'ET_{random.randint(1,3):02d}'
                })
        
        # Generate mixture samples
        mixture_combinations = [
            ['sweet', 'sour'], ['sweet', 'bitter'], ['salty', 'umami'],
            ['sweet', 'umami'], ['sour', 'bitter'], ['salty', 'sour'],
            ['sweet', 'salty', 'umami'], ['bitter', 'astringent']
        ]
        
        for combo in mixture_combinations:
            n_combo_samples = n_mixture_samples // len(mixture_combinations)
            for i in range(n_combo_samples):
                vector = self.generate_mixture(combo)
                data.append(vector)
                labels.append(';'.join(combo))
                metadata.append({
                    'batch_id': f'M{i//50:03d}',
                    'sample_id': f'mix_{i:04d}',
                    'temp_c': np.random.normal(25, 2),
                    'humidity_pct': np.random.normal(45, 5),
                    'instrument_id': f'ET_{random.randint(1,3):02d}'
                })
        
        # Convert to arrays and add sensor effects
        data = np.array(data)
        data = self.add_sensor_effects(data, drift_level, noise_level)
        
        # Create DataFrame
        columns = ['label'] + [f'ch_{i+1}' for i in range(self.n_channels)]
        df_data = np.column_stack([labels, data])
        df = pd.DataFrame(df_data, columns=columns)
        
        # Add metadata
        meta_df = pd.DataFrame(metadata)
        df = pd.concat([df, meta_df], axis=1)
        
        # Convert sensor columns to float
        for i in range(self.n_channels):
            df[f'ch_{i+1}'] = df[f'ch_{i+1}'].astype(float)
            
        return df.sample(frac=1).reset_index(drop=True)  # shuffle
    
    def visualize_data(self, df: pd.DataFrame, save_path: str = None):
        """Create visualization of generated data"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Channel distributions
        sensor_cols = [f'ch_{i+1}' for i in range(self.n_channels)]
        df[sensor_cols].hist(bins=30, ax=axes[0,0])
        axes[0,0].set_title('Sensor Channel Distributions')
        
        # PCA visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(df[sensor_cols])
        
        unique_labels = df['label'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = df['label'] == label
            axes[0,1].scatter(pca_data[mask, 0], pca_data[mask, 1], 
                            c=[color], label=label, alpha=0.6)
        axes[0,1].set_title('PCA Visualization')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Class distribution
        df['label'].value_counts().plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Class Distribution')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Correlation heatmap
        corr_matrix = df[sensor_cols].corr()
        im = axes[1,1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1,1].set_title('Sensor Correlation Matrix')
        plt.colorbar(im, ax=axes[1,1])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Generate and save synthetic e-tongue dataset"""
    generator = ETongueDataGenerator(random_state=42)
    
    print("Generating synthetic e-tongue dataset...")
    df = generator.generate_dataset(
        n_samples_per_class=1200,
        n_mixture_samples=800,
        noise_level=0.05,
        drift_level=0.02
    )
    
    print(f"Generated {len(df)} samples with {df.shape[1]} features")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Save dataset
    df.to_csv('etongue_dataset.csv', index=False)
    print("Dataset saved as 'etongue_dataset.csv'")
    
    # Create visualizations
    generator.visualize_data(df, 'data_visualization.png')
    
    return df

if __name__ == "__main__":
    dataset = main()