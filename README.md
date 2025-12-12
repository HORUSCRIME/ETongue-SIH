# 18-Channel E-Tongue ML System

A complete machine learning system for classifying food taste attributes using an 18-channel electronic tongue sensor. The system supports multi-label classification for sweet, salty, sour, bitter, umami, astringent, and bland tastes.

## Features

- **Synthetic Data Generation**: Realistic sensor data with noise, drift, and multi-label mixtures
- **Multiple ML Models**: MLP (PyTorch), LightGBM, XGBoost, Random Forest
- **Comprehensive Preprocessing**: Scaling, outlier detection, drift correction
- **Robust Evaluation**: Multi-label metrics, calibration curves, feature importance
- **Production API**: FastAPI deployment with health checks and batch processing
- **ONNX Export**: Model export for cross-platform inference

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

```bash
python generate_data.py
```

This creates `etongue_dataset.csv` with realistic 18-channel sensor data.

### 3. Train All Models

```bash
python train.py
```

Trains MLP, LightGBM, XGBoost, and Random Forest models. Saves trained models and preprocessor.

### 4. Evaluate Models

```bash
python eval.py
```

Generates comprehensive evaluation metrics, confusion matrices, and calibration curves.

### 5. Deploy API

```bash
python deploy_api.py
```

Starts FastAPI server on `http://localhost:8000`

### 6. Test API

```bash
python inference_client.py --test --benchmark
```

## File Structure

```
├── generate_data.py      # Synthetic data generator
├── preprocess.py         # Data preprocessing pipeline
├── mlp_model.py         # PyTorch MLP implementation
├── train.py             # Model training script
├── eval.py              # Model evaluation framework
├── deploy_api.py        # FastAPI deployment server
├── inference_client.py  # API client and testing
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Data Format

### Input CSV Format
```csv
label,ch_1,ch_2,...,ch_18,batch_id,sample_id,temp_c,humidity_pct,instrument_id
sweet,0.12,0.85,0.23,...,B001,sweet_0001,25.2,45.1,ET_01
salty;umami,0.45,0.67,0.12,...,B001,mix_0001,24.8,44.9,ET_01
```

### API Request Format
```json
{
  "sample_id": "sample_001",
  "sensors": [0.12, 0.85, 0.23, ..., 0.45],
  "metadata": {"temp_c": 25.0, "humidity_pct": 45.0}
}
```

### API Response Format
```json
{
  "sample_id": "sample_001",
  "probabilities": {
    "sweet": 0.81,
    "salty": 0.02,
    "sour": 0.15,
    "bitter": 0.03,
    "umami": 0.12,
    "astringent": 0.05,
    "bland": 0.08
  },
  "predicted_labels": ["sweet"],
  "confidence_score": 0.81,
  "processing_time_ms": 12.5,
  "model_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Model Architecture

### MLP Model
- **Input**: 18 sensor channels
- **Hidden layers**: [64, 128, 64] with ReLU activation
- **Output**: 7 taste classes (multi-label)
- **Loss**: BCEWithLogitsLoss
- **Regularization**: Dropout (0.3) + Weight Decay (1e-4)

### Alternative Models
- **LightGBM**: One-vs-rest multi-label classification
- **XGBoost**: Separate binary classifiers per taste
- **Random Forest**: MultiOutputClassifier wrapper

## Preprocessing Pipeline

1. **Missing Value Handling**: Median imputation
2. **Outlier Detection**: Z-score based (threshold=3.0)
3. **Drift Correction**: Rolling mean normalization (optional)
4. **Scaling**: StandardScaler, MinMaxScaler, or RobustScaler
5. **Label Encoding**: Multi-label binary matrix

## Evaluation Metrics

- **Exact Match Accuracy**: All labels must match exactly
- **Hamming Accuracy**: Element-wise accuracy
- **Macro/Micro F1**: Averaged across classes/samples
- **ROC-AUC**: Per-class area under curve
- **Calibration**: Reliability diagrams
- **Feature Importance**: SHAP values

## API Endpoints

- `GET /health` - Health check and status
- `POST /predict` - Single sample prediction
- `POST /predict/batch` - Batch prediction (max 100 samples)
- `GET /model/info` - Model information
- `POST /model/reload` - Reload model
- `POST /model/export/onnx` - Export to ONNX format

## Performance Benchmarks

Typical performance on modern hardware:
- **Single prediction**: ~10-15ms
- **Batch processing**: ~2-5ms per sample
- **Throughput**: 200-500 predictions/second

## Synthetic Data Generator

The generator creates realistic sensor patterns:

```python
from generate_data import ETongueDataGenerator

generator = ETongueDataGenerator()
df = generator.generate_dataset(
    n_samples_per_class=1200,
    n_mixture_samples=800,
    noise_level=0.05,
    drift_level=0.02
)
```

### Taste-Specific Patterns
- **Sweet**: High response in channels 3, 5, 9
- **Salty**: Dominant signals in channels 2, 7
- **Sour**: Elevated channels 12, 13
- **Bitter**: High variance in channel 15
- **Umami**: Strong channels 1, 4
- **Astringent**: Active channels 16, 17
- **Bland**: Low signal across all channels

## Error Analysis

### Drift Detection
- Monitor sensor statistics (mean, std, range)
- Flag potential calibration issues
- Automatic drift correction in preprocessing

### Robustness Features
- Outlier detection and removal
- Cross-validation with stratified splits
- Early stopping to prevent overfitting
- Model ensemble capabilities

## Deployment Options

### Local Development
```bash
python deploy_api.py
```

### Production Deployment
```bash
uvicorn deploy_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "deploy_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Continual Learning

The system supports model updates:
1. Collect new labeled data
2. Retrain models with combined dataset
3. Validate performance on holdout set
4. Deploy updated model via API reload

## Troubleshooting

### Common Issues

1. **Model not loading**: Check file paths and dependencies
2. **Poor predictions**: Verify sensor calibration and data quality
3. **API errors**: Check input format and value ranges
4. **Performance issues**: Consider batch processing for multiple samples

### Sensor Validation
- Values should be in [0, 1] range after normalization
- Check for NaN or infinite values
- Monitor for sensor drift or saturation

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## Citation

If you use this system in research, please cite:

```bibtex
@software{etongue_ml_system,
  title={18-Channel E-Tongue ML System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/etongue-ml}
}
```