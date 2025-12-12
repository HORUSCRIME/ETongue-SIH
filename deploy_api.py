from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import torch
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
import logging
import os
from datetime import datetime
import uuid

from mlp_model import ETongueMLP
from preprocess import ETonguePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorReading(BaseModel):
    sample_id: str = Field(..., description="Unique sample identifier")
    sensors: List[float] = Field(..., min_items=18, max_items=18, 
                                description="18 sensor channel readings")
    metadata: Optional[Dict[str, Any]] = Field(default={}, 
                                             description="Additional metadata")

class TastePrediction(BaseModel):
    sample_id: str
    probabilities: Dict[str, float]
    predicted_labels: List[str]
    confidence_score: float
    processing_time_ms: float
    model_version: str
    timestamp: str

class BatchSensorReading(BaseModel):
    samples: List[SensorReading]

class BatchTastePrediction(BaseModel):
    predictions: List[TastePrediction]
    batch_id: str
    total_samples: int
    processing_time_ms: float

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float

class ETonguePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.class_names = None
        self.model_version = "1.0.0"
        self.start_time = datetime.now()
        self.prediction_count = 0
        
    def load_models(self):
        try:
            metadata = joblib.load('etongue_metadata.pkl')
            self.class_names = metadata['class_names']
            
            self.preprocessor = joblib.load('etongue_preprocessor.pkl')
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = ETongueMLP(
                input_dim=metadata['input_dim'],
                output_dim=metadata['output_dim']
            )
            self.model.load_state_dict(torch.load('etongue_mlp.pth', map_location=device))
            self.model.eval()
            self.model.to(device)
            
            logger.info(f"Models loaded successfully on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def validate_sensor_data(self, sensors: List[float]) -> bool:
        if len(sensors) != 18:
            return False
        
        if any(s < 0 or s > 1 for s in sensors):
            logger.warning("Sensor values outside expected range [0,1]")
        
        if any(not np.isfinite(s) for s in sensors):
            return False
            
        return True
    
    def detect_drift(self, sensors: List[float]) -> Dict[str, Any]:
        sensors_array = np.array(sensors)
        
        mean_signal = np.mean(sensors_array)
        std_signal = np.std(sensors_array)
        max_signal = np.max(sensors_array)
        min_signal = np.min(sensors_array)
        
        drift_indicators = {
            'mean_in_range': 0.05 <= mean_signal <= 0.95,
            'std_reasonable': std_signal > 0.01,  # Some variation expected
            'range_reasonable': (max_signal - min_signal) > 0.02,
            'no_saturation': max_signal < 0.98 and min_signal > 0.02
        }
        
        drift_detected = not all(drift_indicators.values())
        
        return {
            'drift_detected': drift_detected,
            'indicators': drift_indicators,
            'statistics': {
                'mean': float(mean_signal),
                'std': float(std_signal),
                'min': float(min_signal),
                'max': float(max_signal)
            }
        }
    
    def predict_single(self, sensor_reading: SensorReading) -> TastePrediction:
        start_time = datetime.now()
        
        if not self.validate_sensor_data(sensor_reading.sensors):
            raise ValueError("Invalid sensor data")
        
        drift_info = self.detect_drift(sensor_reading.sensors)
        if drift_info['drift_detected']:
            logger.warning(f"Potential sensor drift detected for {sensor_reading.sample_id}")
        
        df = pd.DataFrame([sensor_reading.sensors], 
                         columns=[f'ch_{i+1}' for i in range(18)])
        
        X = self.preprocessor.transform(df, apply_drift_correction=False)
        
        device = next(self.model.parameters()).device
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        prob_dict = {name: float(prob) for name, prob in zip(self.class_names, probabilities)}
        
        predicted_labels = [name for name, prob in prob_dict.items() if prob > 0.5]
        if not predicted_labels:
            predicted_labels = ['bland']  
        confidence_score = float(np.max(probabilities))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        self.prediction_count += 1
        
        return TastePrediction(
            sample_id=sensor_reading.sample_id,
            probabilities=prob_dict,
            predicted_labels=predicted_labels,
            confidence_score=confidence_score,
            processing_time_ms=processing_time,
            model_version=self.model_version,
            timestamp=datetime.now().isoformat()
        )
    
    def predict_batch(self, batch_reading: BatchSensorReading) -> BatchTastePrediction:
        start_time = datetime.now()
        batch_id = str(uuid.uuid4())
        
        predictions = []
        for sample in batch_reading.samples:
            try:
                prediction = self.predict_single(sample)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict sample {sample.sample_id}: {e}")
                error_prediction = TastePrediction(
                    sample_id=sample.sample_id,
                    probabilities={name: 0.0 for name in self.class_names},
                    predicted_labels=['error'],
                    confidence_score=0.0,
                    processing_time_ms=0.0,
                    model_version=self.model_version,
                    timestamp=datetime.now().isoformat()
                )
                predictions.append(error_prediction)
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchTastePrediction(
            predictions=predictions,
            batch_id=batch_id,
            total_samples=len(batch_reading.samples),
            processing_time_ms=total_time
        )

predictor = ETonguePredictor()

app = FastAPI(
    title="E-Tongue Taste Classification API",
    description="API for classifying food taste attributes using 18-channel electronic tongue sensor",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    success = predictor.load_models()
    if not success:
        logger.error("Failed to load models on startup")

@app.get("/health", response_model=HealthCheck)
async def health_check():
    uptime = (datetime.now() - predictor.start_time).total_seconds()
    
    return HealthCheck(
        status="healthy" if predictor.model is not None else "unhealthy",
        model_loaded=predictor.model is not None,
        version=predictor.model_version,
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=TastePrediction)
async def predict_taste(sensor_reading: SensorReading):
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prediction = predictor.predict_single(sensor_reading)
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=BatchTastePrediction)
async def predict_taste_batch(batch_reading: BatchSensorReading):
    if predictor.model is not None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(batch_reading.samples) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    try:
        predictions = predictor.predict_batch(batch_reading)
        return predictions
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/info")
async def model_info():
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": predictor.model_version,
        "class_names": predictor.class_names,
        "input_channels": 18,
        "prediction_count": predictor.prediction_count,
        "uptime_seconds": (datetime.now() - predictor.start_time).total_seconds()
    }

@app.post("/model/reload")
async def reload_model():
    success = predictor.load_models()
    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.post("/model/export/onnx")
async def export_onnx():
    """Export model to ONNX format"""
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        dummy_input = torch.randn(1, 18)
        
        torch.onnx.export(
            predictor.model,
            dummy_input,
            "etongue_model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['sensor_inputs'],
            output_names=['taste_logits'],
            dynamic_axes={
                'sensor_inputs': {0: 'batch_size'},
                'taste_logits': {0: 'batch_size'}
            }
        )
        
        return {"status": "success", "message": "Model exported to etongue_model.onnx"}
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
