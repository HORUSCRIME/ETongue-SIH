import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
import shap
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from mlp_model import ETongueMLP
from preprocess import ETonguePreprocessor

class ETongueEvaluator:
    def __init__(self, data_dict: Dict, class_names: List[str]):
        self.data_dict = data_dict
        self.class_names = class_names
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_models(self):
        models = {}
        
        try:
            metadata = joblib.load('etongue_metadata.pkl')
            mlp_model = ETongueMLP(
                input_dim=metadata['input_dim'],
                output_dim=metadata['output_dim']
            )
            mlp_model.load_state_dict(torch.load('etongue_mlp.pth', map_location=self.device))
            mlp_model.eval()
            models['mlp'] = mlp_model
        except:
            print("Could not load MLP model")
        
        try:
            models['lightgbm'] = joblib.load('etongue_lightgbm.pkl')
        except:
            print("Could not load LightGBM model")
            
        try:
            models['random_forest'] = joblib.load('etongue_rf.pkl')
        except:
            print("Could not load Random Forest model")
            
        try:
            models['xgboost'] = joblib.load('etongue_xgb.pkl')
        except:
            print("Could not load XGBoost model")
        
        return models
    
    def predict_mlp(self, model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(model(X_tensor)).cpu().numpy()
            preds = (probs > 0.5).astype(int)
        return preds, probs
    
    def predict_lightgbm(self, models: List, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = np.zeros((X.shape[0], len(models)))
        for i, model in enumerate(models):
            probs[:, i] = model.predict(X, num_iteration=model.best_iteration)
        preds = (probs > 0.5).astype(int)
        return preds, probs
    
    def predict_sklearn(self, model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = model.predict(X)
        probs = model.predict_proba(X)
        if isinstance(probs, list):
            probs = np.column_stack([p[:, 1] for p in probs])
        return preds, probs
    
    def predict_xgboost(self, models: List, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = np.zeros((X.shape[0], len(models)))
        for i, model in enumerate(models):
            probs[:, i] = model.predict_proba(X)[:, 1]
        preds = (probs > 0.5).astype(int)
        return preds, probs
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_probs: np.ndarray) -> Dict[str, Any]:
        metrics = {}
        
        exact_match = np.all(y_true == y_pred, axis=1)
        metrics['exact_match_accuracy'] = np.mean(exact_match)
        
        metrics['hamming_accuracy'] = accuracy_score(y_true, y_pred)
        
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['micro_recall'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class_f1'] = dict(zip(self.class_names, per_class_f1))
        metrics['per_class_precision'] = dict(zip(self.class_names, per_class_precision))
        metrics['per_class_recall'] = dict(zip(self.class_names, per_class_recall))
        
        try:
            roc_auc_per_class = []
            for i in range(y_true.shape[1]):
                if len(np.unique(y_true[:, i])) > 1:  
                    auc = roc_auc_score(y_true[:, i], y_probs[:, i])
                    roc_auc_per_class.append(auc)
                else:
                    roc_auc_per_class.append(0.5)  
            
            metrics['roc_auc_macro'] = np.mean(roc_auc_per_class)
            metrics['per_class_roc_auc'] = dict(zip(self.class_names, roc_auc_per_class))
        except:
            metrics['roc_auc_macro'] = 0.0
            metrics['per_class_roc_auc'] = {name: 0.0 for name in self.class_names}
        
        return metrics
    
    def plot_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str, save_path: str = None):
        n_classes = len(self.class_names)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], 
                       xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            axes[i].set_title(f'{class_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        if len(self.class_names) < len(axes):
            axes[-1].set_visible(False)
        
        plt.suptitle(f'Confusion Matrices - {model_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curves(self, y_true: np.ndarray, y_probs: np.ndarray,
                               model_name: str, save_path: str = None):
        n_classes = len(self.class_names)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            if len(np.unique(y_true[:, i])) > 1:
                fraction_pos, mean_pred = calibration_curve(
                    y_true[:, i], y_probs[:, i], n_bins=10
                )
                
                axes[i].plot(mean_pred, fraction_pos, 's-', label=f'{class_name}')
                axes[i].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                axes[i].set_xlabel('Mean Predicted Probability')
                axes[i].set_ylabel('Fraction of Positives')
                axes[i].set_title(f'{class_name}')
                axes[i].legend()
                axes[i].grid(True)
            else:
                axes[i].text(0.5, 0.5, 'Single class\nin test set', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{class_name}')
        
        if len(self.class_names) < len(axes):
            axes[-1].set_visible(False)
        
        plt.suptitle(f'Calibration Curves - {model_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, model, model_type: str, X_test: np.ndarray):
        try:
            if model_type == 'mlp':
                explainer = shap.DeepExplainer(model, torch.FloatTensor(X_test[:100]))
                shap_values = explainer.shap_values(torch.FloatTensor(X_test[:100]))
                
                if isinstance(shap_values, list):
                    importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    importance = np.abs(shap_values).mean(axis=0)
                    
            elif model_type == 'random_forest':
                importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
                
            else:
                if hasattr(model, '__iter__'): 
                    importance_list = []
                    for m in model:
                        explainer = shap.TreeExplainer(m)
                        shap_vals = explainer.shap_values(X_test[:100])
                        importance_list.append(np.abs(shap_vals).mean(axis=0))
                    importance = np.mean(importance_list, axis=0)
                else:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test[:100])
                    importance = np.abs(shap_values).mean(axis=0)
            
            return importance
            
        except Exception as e:
            print(f"Could not calculate feature importance for {model_type}: {e}")
            return np.zeros(X_test.shape[1])
    
    def pca_visualization(self, X_test: np.ndarray, y_test: np.ndarray, 
                         save_path: str = None):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)
        
        plt.figure(figsize=(12, 8))
        
        combined_labels = []
        for i in range(len(y_test)):
            active_classes = [self.class_names[j] for j in range(len(self.class_names)) 
                            if y_test[i, j] == 1]
            if active_classes:
                combined_labels.append(';'.join(active_classes))
            else:
                combined_labels.append('none')
        
        unique_labels = list(set(combined_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = np.array(combined_labels) == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[color], label=label, alpha=0.6)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA Visualization of Test Data')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_all_models(self):
        models = self.load_models()
        results = {}
        
        X_test = self.data_dict['X_test']
        y_test = self.data_dict['y_test']
        
        print("Evaluating all models...")
        print("=" * 50)
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name.upper()}...")
            
            if model_name == 'mlp':
                y_pred, y_probs = self.predict_mlp(model, X_test)
            elif model_name == 'lightgbm':
                y_pred, y_probs = self.predict_lightgbm(model, X_test)
            elif model_name == 'random_forest':
                y_pred, y_probs = self.predict_sklearn(model, X_test)
            elif model_name == 'xgboost':
                y_pred, y_probs = self.predict_xgboost(model, X_test)
            
            metrics = self.calculate_metrics(y_test, y_pred, y_probs)
            results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_probs
            }
            
            print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
            print(f"Hamming Accuracy: {metrics['hamming_accuracy']:.3f}")
            print(f"Macro F1: {metrics['macro_f1']:.3f}")
            print(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.3f}")
            
            self.plot_confusion_matrices(y_test, y_pred, model_name, 
                                       f'{model_name}_confusion_matrices.png')
            self.plot_calibration_curves(y_test, y_probs, model_name,
                                       f'{model_name}_calibration.png')
            
            importance = self.analyze_feature_importance(model, model_name, X_test)
            results[model_name]['feature_importance'] = importance
        
        self.pca_visualization(X_test, y_test, 'pca_visualization.png')
        
        self.create_comparison_table(results)
        
        return results
    
    def create_comparison_table(self, results: Dict):
        comparison_data = []
        
        for model_name, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name.upper(),
                'Exact Match Acc': f"{metrics['exact_match_accuracy']:.3f}",
                'Hamming Acc': f"{metrics['hamming_accuracy']:.3f}",
                'Macro F1': f"{metrics['macro_f1']:.3f}",
                'Macro Precision': f"{metrics['macro_precision']:.3f}",
                'Macro Recall': f"{metrics['macro_recall']:.3f}",
                'ROC-AUC': f"{metrics['roc_auc_macro']:.3f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(df_comparison.to_string(index=False))
        
        df_comparison.to_csv('model_comparison.csv', index=False)
        print("\nComparison saved as 'model_comparison.csv'")

def main():
    print("E-Tongue Model Evaluation")
    print("=" * 50)
    
    try:
        data_dict = {
            'X_test': np.load('X_test.npy'),
            'y_test': np.load('y_test.npy')
        }
        class_names = joblib.load('etongue_metadata.pkl')['class_names']
    except:
        print("Test data not found. Running training first...")
        from train import main as train_main
        trainer, data_dict = train_main()
        class_names = data_dict['class_names']
        
        np.save('X_test.npy', data_dict['X_test'])
        np.save('y_test.npy', data_dict['y_test'])
    
    evaluator = ETongueEvaluator(data_dict, class_names)
    
    results = evaluator.evaluate_all_models()
    
    print("\nEvaluation completed!")
    print("Generated files:")
    print("- *_confusion_matrices.png (Confusion matrices for each model)")
    print("- *_calibration.png (Calibration curves for each model)")
    print("- pca_visualization.png (PCA visualization)")
    print("- model_comparison.csv (Performance comparison)")
    
    return evaluator, results

if __name__ == "__main__":
    evaluator, results = main()
