import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict

class ETongueDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ETongueMLP(nn.Module):
    def __init__(self, input_dim=18, hidden_dims=[64, 128, 64], 
                 output_dim=7, dropout_rate=0.3):
        super(ETongueMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
    
    def predict(self, x, threshold=0.5):
        probs = self.predict_proba(x)
        return (probs > threshold).float()

class ETongueTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        f1_scores = []
        for i in range(all_targets.shape[1]):
            tp = np.sum((all_preds[:, i] == 1) & (all_targets[:, i] == 1))
            fp = np.sum((all_preds[:, i] == 1) & (all_targets[:, i] == 0))
            fn = np.sum((all_preds[:, i] == 0) & (all_targets[:, i] == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        macro_f1 = np.mean(f1_scores)
        
        return total_loss / len(val_loader), macro_f1
    
    def fit(self, train_loader, val_loader, epochs=100, lr=0.001, 
            weight_decay=1e-4, patience=10, min_delta=1e-4):
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                   lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            val_loss, val_f1 = self.validate(val_loader, criterion)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return self.history

def create_data_loaders(data_dict: Dict, batch_size=32, num_workers=0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_dataset = ETongueDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = ETongueDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = ETongueDataset(data_dict['X_test'], data_dict['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def main():
    from preprocess import prepare_data
    from generate_data import ETongueDataGenerator
    
    generator = ETongueDataGenerator()
    df = generator.generate_dataset(n_samples_per_class=300, n_mixture_samples=150)
    df.to_csv('test_mlp_data.csv', index=False)
    
    data_dict = prepare_data('test_mlp_data.csv')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ETongueMLP(input_dim=18, hidden_dims=[64, 128, 64], output_dim=7)
    
    train_loader, val_loader, test_loader = create_data_loaders(data_dict, batch_size=32)
    
    trainer = ETongueTrainer(model, device)
    history = trainer.fit(train_loader, val_loader, epochs=50, lr=0.001)
    
    model.eval()
    with torch.no_grad():
        sample_x = torch.FloatTensor(data_dict['X_test'][:5]).to(device)
        probs = model.predict_proba(sample_x)
        preds = model.predict(sample_x)
        
    print("Sample predictions:")
    print("Probabilities:", probs.cpu().numpy())
    print("Binary predictions:", preds.cpu().numpy())
    
    return model, trainer, history

if __name__ == "__main__":
    model, trainer, history = main()
