# E-Tongue ML System - Step-by-Step Deployment Guide

This guide walks you through setting up and deploying the complete 18-channel e-tongue ML system from scratch.

## Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- GPU optional (CUDA-compatible for faster training)

## Step 1: Environment Setup

### 1.1 Create Virtual Environment
```bash
# Create virtual environment
python -m venv etongue_env

# Activate environment
# Windows:
etongue_env\Scripts\activate
# Linux/Mac:
source etongue_env/bin/activate
```

### 1.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 1.3 Verify Installation
```bash
python -c "import torch, sklearn, pandas, fastapi; print('All dependencies installed successfully')"
```

## Step 2: Data Generation and Preparation

### 2.1 Generate Synthetic Dataset
```bash
python generate_data.py
```

**Expected Output:**
- `etongue_dataset.csv` (8,000+ samples)
- `data_visualization.png` (data analysis plots)
- Console output showing class distribution

**Verification:**
```bash
python -c "import pandas as pd; df = pd.read_csv('etongue_dataset.csv'); print(f'Dataset shape: {df.shape}'); print(f'Classes: {df.label.value_counts()}')"
```

### 2.2 Test Preprocessing
```bash
python preprocess.py
```

**Expected Output:**
- Preprocessing pipeline validation
- Sample data transformations
- Outlier detection statistics

## Step 3: Model Training

### 3.1 Train All Models
```bash
python train.py
```

**Expected Output:**
- `etongue_mlp.pth` (PyTorch MLP model)
- `etongue_lightgbm.pkl` (LightGBM models)
- `etongue_rf.pkl` (Random Forest model)
- `etongue_xgb.pkl` (XGBoost models)
- `etongue_preprocessor.pkl` (Data preprocessor)
- `etongue_metadata.pkl` (Model metadata)
- `training_history.png` (Training curves)

**Training Time:** 5-15 minutes depending on hardware

**Verification:**
```bash
# Check if all model files exist
python -c "import os; files=['etongue_mlp.pth', 'etongue_lightgbm.pkl', 'etongue_rf.pkl', 'etongue_xgb.pkl', 'etongue_preprocessor.pkl', 'etongue_metadata.pkl']; print('Missing files:', [f for f in files if not os.path.exists(f)])"
```

## Step 4: Model Evaluation

### 4.1 Run Comprehensive Evaluation
```bash
python eval.py
```

**Expected Output:**
- Model performance comparison table
- Confusion matrices for each model
- Calibration curves
- PCA visualization
- `model_comparison.csv`
- Multiple PNG visualization files

**Key Metrics to Check:**
- Exact Match Accuracy > 0.7
- Macro F1 Score > 0.8
- ROC-AUC > 0.9

## Step 5: API Deployment

### 5.1 Start API Server
```bash
python deploy_api.py
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Models loaded successfully on cpu
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5.2 Verify API Health
Open new terminal and run:
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime_seconds": 10.5
}
```

## Step 6: API Testing

### 6.1 Run Functionality Tests
```bash
python inference_client.py --test
```

**Expected Output:**
- ✅ Health check passed
- ✅ Model info retrieved
- ✅ Single prediction successful
- ✅ Batch prediction successful

### 6.2 Run Performance Benchmark
```bash
python inference_client.py --benchmark --samples 50
```

**Expected Performance:**
- Single predictions: 10-20ms response time
- Batch processing: 2-8ms per sample
- Throughput: 100-500 predictions/second

### 6.3 Interactive Testing
```bash
python inference_client.py
```

Enter test sensor readings when prompted:
```
Sensor readings: 0.1,0.2,0.8,0.1,0.7,0.2,0.1,0.1,0.6,0.4,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1
```

## Step 7: Production Deployment

### 7.1 Production Server Setup
```bash
# Install production server
pip install gunicorn

# Start with multiple workers
gunicorn deploy_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 7.2 Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "deploy_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t etongue-api .
docker run -p 8000:8000 etongue-api
```

### 7.3 NGINX Reverse Proxy (Optional)

Create `/etc/nginx/sites-available/etongue`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Step 8: Monitoring and Maintenance

### 8.1 Health Monitoring
Set up periodic health checks:
```bash
# Add to crontab for monitoring
*/5 * * * * curl -f http://localhost:8000/health || echo "API down" | mail admin@company.com
```

### 8.2 Log Monitoring
Monitor API logs for errors:
```bash
tail -f /var/log/etongue-api.log
```

### 8.3 Model Updates
To update models:
1. Retrain with new data: `python train.py`
2. Reload via API: `curl -X POST http://localhost:8000/model/reload`

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

#### 2. CUDA Issues
```bash
# Error: CUDA out of memory
# Solution: Use CPU or reduce batch size
export CUDA_VISIBLE_DEVICES=""
```

#### 3. Model Loading Errors
```bash
# Error: Model files not found
# Solution: Ensure training completed successfully
python train.py
```

#### 4. API Connection Issues
```bash
# Error: Connection refused
# Solution: Check if API server is running
ps aux | grep deploy_api
```

#### 5. Poor Model Performance
```bash
# Check data quality
python -c "import pandas as pd; df = pd.read_csv('etongue_dataset.csv'); print(df.describe())"

# Retrain with more data
python generate_data.py  # Generate more samples
python train.py          # Retrain models
```

### Performance Optimization

#### 1. GPU Acceleration
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 2. Batch Processing
Use batch endpoints for multiple samples:
```python
# Instead of multiple single requests
for sample in samples:
    predict_single(sample)

# Use batch request
predict_batch(samples)
```

#### 3. Model Optimization
```bash
# Export to ONNX for faster inference
curl -X POST http://localhost:8000/model/export/onnx
```

## Validation Checklist

Before production deployment, verify:

- [ ] All model files exist and load successfully
- [ ] API health check returns "healthy"
- [ ] Single prediction works with test data
- [ ] Batch prediction handles multiple samples
- [ ] Response times are acceptable (< 100ms for single prediction)
- [ ] Model accuracy meets requirements (> 80% F1 score)
- [ ] Error handling works for invalid inputs
- [ ] Logging is configured properly
- [ ] Security measures are in place (if needed)

## Support and Maintenance

### Regular Tasks
1. **Weekly**: Check API health and performance metrics
2. **Monthly**: Review prediction logs for data drift
3. **Quarterly**: Retrain models with new data
4. **Annually**: Update dependencies and security patches

### Backup Strategy
```bash
# Backup trained models
tar -czf etongue_models_$(date +%Y%m%d).tar.gz *.pth *.pkl

# Backup configuration
cp requirements.txt config_backup_$(date +%Y%m%d).txt
```

### Contact Information
For technical support or questions:
- Documentation: See README.md
- Issues: Check troubleshooting section
- Updates: Monitor for new releases

---

**Deployment Complete!** 🎉

Your e-tongue ML system is now ready for production use. The API is accessible at `http://localhost:8000` with full documentation at `http://localhost:8000/docs`.