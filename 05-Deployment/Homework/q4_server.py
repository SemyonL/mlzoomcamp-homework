from fastapi import FastAPI
import uvicorn
import pickle
from typing import Dict, Any

app = FastAPI(title="Model servicing")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post('/predict')
def predict(client: Dict[str, Any]):
    result = float(pipeline.predict_proba(client)[0,1])
    return {
        "subscription_probability": result,
        "subscription": bool(result >= 0.5)
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)