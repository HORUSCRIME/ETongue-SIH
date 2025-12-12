import torch
import joblib
import numpy as np
import pandas as pd
from mlp_model import ETongueMLP

class TastePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.class_names = None
        self.load_models()
    
    def load_models(self):
        """Load trained model and preprocessor"""
        # Load metadata
        metadata = joblib.load('etongue_metadata.pkl')
        self.class_names = metadata['class_names']
        
        # Load preprocessor
        self.preprocessor = joblib.load('etongue_preprocessor.pkl')
        
        # Load MLP model
        self.model = ETongueMLP(
            input_dim=metadata['input_dim'],
            output_dim=metadata['output_dim']
        )
        self.model.load_state_dict(torch.load('etongue_mlp.pth', map_location='cpu'))
        self.model.eval()
    
    def predict_taste(self, sensor_readings, threshold=0.5):
        """Predict taste from 18 sensor readings"""
        # Validate input
        if len(sensor_readings) != 18:
            raise ValueError("Must provide exactly 18 sensor readings")
        
        # Prepare data
        df = pd.DataFrame([sensor_readings], columns=[f'ch_{i+1}' for i in range(18)])
        X = self.preprocessor.transform(df, apply_drift_correction=False)
        
        # Predict
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.sigmoid(logits).numpy()[0]
        
        # Get results
        results = {}
        predicted_tastes = []
        
        for i, (taste, prob) in enumerate(zip(self.class_names, probabilities)):
            results[taste] = float(prob)
            if prob > threshold:
                predicted_tastes.append(taste)
        
        if not predicted_tastes:
            predicted_tastes = ['bland']
        
        return {
            'predicted_tastes': predicted_tastes,
            'probabilities': results,
            'confidence': float(max(probabilities))
        }

def main():
    """Interactive taste prediction"""
    predictor = TastePredictor()
    
    print("E-Tongue Taste Predictor")
    print("=" * 30)
    print("Enter 18 sensor readings (0-1 range, comma-separated)")
    print("Example: 0.1,0.2,0.8,0.1,0.7,0.2,0.1,0.1,0.6,0.4,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Sensor readings: ").strip()
            if user_input.lower() == 'quit':
                break
            
            # Parse input
            readings = [float(x.strip()) for x in user_input.split(',')]
            
            # Predict
            result = predictor.predict_taste(readings)
            
            # Display results
            print(f"\n🎯 Predicted Taste(s): {', '.join(result['predicted_tastes'])}")
            print(f"🔥 Confidence: {result['confidence']:.3f}")
            print("\n📊 All Probabilities:")
            for taste, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {taste:12}: {prob:.3f}")
            print("-" * 30)
            
        except ValueError as e:
            print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    main()