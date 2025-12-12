import requests
import json
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any
import argparse

class ETongueClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def predict_single(self, sample_id: str, sensors: List[float], 
                      metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        if metadata is None:
            metadata = {}
            
        payload = {
            "sample_id": sample_id,
            "sensors": sensors,
            "metadata": metadata
        }
        
        try:
            response = requests.post(f"{self.base_url}/predict", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def predict_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {"samples": samples}
        
        try:
            response = requests.post(f"{self.base_url}/predict/batch", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def reload_model(self) -> Dict[str, Any]:
        try:
            response = requests.post(f"{self.base_url}/model/reload")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def export_onnx(self) -> Dict[str, Any]:
        try:
            response = requests.post(f"{self.base_url}/model/export/onnx")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def generate_test_samples(n_samples: int = 5) -> List[Dict[str, Any]]:
    samples = []
    
    taste_patterns = {
        'sweet': [0.1, 0.2, 0.8, 0.1, 0.7, 0.2, 0.1, 0.1, 0.6, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'salty': [0.1, 0.9, 0.2, 0.1, 0.1, 0.1, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'sour': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1],
        'bitter': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1],
        'umami': [0.7, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    }
    
    taste_names = list(taste_patterns.keys())
    
    for i in range(n_samples):
        if np.random.random() < 0.7:  
            taste = np.random.choice(taste_names)
            sensors = taste_patterns[taste].copy()
            expected_label = taste
        else:  
            taste1, taste2 = np.random.choice(taste_names, 2, replace=False)
            sensors = [(a + b) / 2 for a, b in zip(taste_patterns[taste1], taste_patterns[taste2])]
            expected_label = f"{taste1};{taste2}"
        
        sensors = [max(0, min(1, s + np.random.normal(0, 0.05))) for s in sensors]
        
        samples.append({
            "sample_id": f"test_sample_{i+1:03d}",
            "sensors": sensors,
            "metadata": {
                "expected_label": expected_label,
                "temp_c": 25.0 + np.random.normal(0, 1),
                "humidity_pct": 45.0 + np.random.normal(0, 3),
                "instrument_id": "ET_01"
            }
        })
    
    return samples

def test_api_functionality(client: ETongueClient):
    print("Testing E-Tongue API Functionality")
    print("=" * 50)
    
    print("1. Health Check:")
    health = client.health_check()
    if "error" in health:
        print(f"    Health check failed: {health['error']}")
        return False
    else:
        print(f"    Status: {health['status']}")
        print(f"    Model loaded: {health['model_loaded']}")
        print(f"    Uptime: {health['uptime_seconds']:.1f}s")
    
    print("\n2. Model Information:")
    info = client.get_model_info()
    if "error" in info:
        print(f"    Failed to get model info: {info['error']}")
    else:
        print(f"    Version: {info['model_version']}")
        print(f"    Classes: {info['class_names']}")
        print(f"    Predictions made: {info['prediction_count']}")
    
    print("\n3. Single Prediction Test:")
    test_samples = generate_test_samples(1)
    sample = test_samples[0]
    
    start_time = time.time()
    result = client.predict_single(
        sample["sample_id"], 
        sample["sensors"], 
        sample["metadata"]
    )
    end_time = time.time()
    
    if "error" in result:
        print(f"    Single prediction failed: {result['error']}")
    else:
        print(f"    Sample ID: {result['sample_id']}")
        print(f"    Predicted: {result['predicted_labels']}")
        print(f"    Expected: {sample['metadata']['expected_label']}")
        print(f"    Confidence: {result['confidence_score']:.3f}")
        print(f"    API Response time: {(end_time - start_time)*1000:.1f}ms")
        print(f"    Processing time: {result['processing_time_ms']:.1f}ms")
        
        probs = result['probabilities']
        top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        print("   Top probabilities:")
        for taste, prob in top_probs:
            print(f"      {taste}: {prob:.3f}")
    
    print("\n4. Batch Prediction Test:")
    batch_samples = generate_test_samples(5)
    
    start_time = time.time()
    batch_result = client.predict_batch(batch_samples)
    end_time = time.time()
    
    if "error" in batch_result:
        print(f"    Batch prediction failed: {batch_result['error']}")
    else:
        print(f"    Batch ID: {batch_result['batch_id']}")
        print(f"    Total samples: {batch_result['total_samples']}")
        print(f"    API Response time: {(end_time - start_time)*1000:.1f}ms")
        print(f"    Processing time: {batch_result['processing_time_ms']:.1f}ms")
        print(f"    Avg time per sample: {batch_result['processing_time_ms']/batch_result['total_samples']:.1f}ms")
        
        correct_predictions = 0
        for i, pred in enumerate(batch_result['predictions']):
            expected = batch_samples[i]['metadata']['expected_label']
            predicted = ';'.join(pred['predicted_labels'])
            if expected == predicted:
                correct_predictions += 1
            print(f"      Sample {i+1}: Expected={expected}, Predicted={predicted}")
        
        accuracy = correct_predictions / len(batch_result['predictions'])
        print(f"    Batch accuracy: {accuracy:.1%}")
    
    return True

def benchmark_performance(client: ETongueClient, n_samples: int = 100):
    print(f"\nBenchmarking Performance ({n_samples} samples)")
    print("=" * 50)
    
    samples = generate_test_samples(n_samples)
    
    print("Single Predictions:")
    single_times = []
    for i, sample in enumerate(samples[:10]):  
        start_time = time.time()
        result = client.predict_single(sample["sample_id"], sample["sensors"])
        end_time = time.time()
        
        if "error" not in result:
            single_times.append((end_time - start_time) * 1000)
        
        if (i + 1) % 5 == 0:
            print(f"   Completed {i+1}/10 predictions")
    
    if single_times:
        print(f"    Average response time: {np.mean(single_times):.1f}ms")
        print(f"    Min response time: {np.min(single_times):.1f}ms")
        print(f"    Max response time: {np.max(single_times):.1f}ms")
        print(f"    Throughput: {1000/np.mean(single_times):.1f} predictions/second")
    
    print("\nBatch Predictions:")
    batch_sizes = [5, 10, 20, 50]
    
    for batch_size in batch_sizes:
        if batch_size <= len(samples):
            batch_samples = samples[:batch_size]
            
            start_time = time.time()
            result = client.predict_batch(batch_samples)
            end_time = time.time()
            
            if "error" not in result:
                total_time = (end_time - start_time) * 1000
                per_sample_time = total_time / batch_size
                throughput = batch_size / (total_time / 1000)
                
                print(f"   Batch size {batch_size:2d}: {total_time:6.1f}ms total, "
                      f"{per_sample_time:5.1f}ms/sample, {throughput:5.1f} samples/sec")

def main():
    parser = argparse.ArgumentParser(description="E-Tongue API Client")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="API base URL")
    parser.add_argument("--test", action="store_true", 
                       help="Run functionality tests")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run performance benchmark")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples for benchmark")
    
    args = parser.parse_args()
    
    client = ETongueClient(args.url)
    
    health = client.health_check()
    if "error" in health:
        print(f" Cannot connect to API at {args.url}")
        print(f"Error: {health['error']}")
        print("\nMake sure the API server is running:")
        print("python deploy_api.py")
        return
    
    print(f" Connected to E-Tongue API at {args.url}")
    
    if args.test:
        success = test_api_functionality(client)
        if not success:
            return
    
    if args.benchmark:
        benchmark_performance(client, args.samples)
    
    if not args.test and not args.benchmark:
        print("\nInteractive Mode - Enter sensor readings:")
        print("Format: 18 comma-separated values (0-1 range)")
        print("Type 'quit' to exit")
        
        while True:
            try:
                user_input = input("\nSensor readings: ").strip()
                if user_input.lower() == 'quit':
                    break
                
                sensors = [float(x.strip()) for x in user_input.split(',')]
                if len(sensors) != 18:
                    print(" Please provide exactly 18 sensor values")
                    continue
                
                sample_id = f"interactive_{int(time.time())}"
                result = client.predict_single(sample_id, sensors)
                
                if "error" in result:
                    print(f" Prediction failed: {result['error']}")
                else:
                    print(f" Predicted tastes: {', '.join(result['predicted_labels'])}")
                    print(f" Confidence: {result['confidence_score']:.3f}")
                    
                    probs = result['probabilities']
                    print("Probabilities:")
                    for taste, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {taste}: {prob:.3f}")
                        
            except KeyboardInterrupt:
                break
            except ValueError:
                print(" Invalid input format. Use comma-separated numbers.")
            except Exception as e:
                print(f" Error: {e}")
        
        print("Goodbye!")

if __name__ == "__main__":
    main()
