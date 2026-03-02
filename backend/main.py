from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import pickle
import numpy as np
import Orange
import os
import json

app = FastAPI(title="Orange Model Predictor API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_models"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store loaded models in memory (basic caching for single user/demo)
loaded_models = {}

@app.get("/")
def read_root():
    return {"message": "Orange Model Predictor API is running."}

@app.post("/api/upload")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".pkcls"):
        raise HTTPException(status_code=400, detail="Only .pkcls files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save file
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    # Load and parse model
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
            
        # Extract features (attributes)
        features = []
        for attr in model.domain.attributes:
            feature_info = {
                "name": attr.name,
                "type": "continuous" if attr.is_continuous else ("discrete" if attr.is_discrete else "other")
            }
            if attr.is_discrete:
                feature_info["values"] = attr.values
            features.append(feature_info)
            
        # Extract class variable info
        class_var = model.domain.class_var
        class_info = {
            "name": class_var.name,
            "values": class_var.values
        }
        
        # Cache the model
        loaded_models[file.filename] = model
        
        return JSONResponse(content={
            "filename": file.filename,
            "features": features,
            "class_variable": class_info,
            "message": "Model loaded successfully."
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse Orange model: {str(e)}")


@app.post("/api/predict")
async def predict(filename: str = Form(...), features_json: str = Form(...)):
    # features_json should be a JSON string of a dictionary: {"sepal length": 5.1, "sepal width": 3.5, ...}
    try:
        input_features = json.loads(features_json)
    except json.JSONDecodeError:
         raise HTTPException(status_code=400, detail="Invalid JSON for features.")

    if filename not in loaded_models:
        # Try to load it from disk
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Model not found. Please upload it first.")
        try:
             with open(file_path, "rb") as f:
                 loaded_models[filename] = pickle.load(f)
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Failed to load model from disk: {str(e)}")
             
    model = loaded_models[filename]
    
    # Construct data row based on domain attributes order
    try:
        data_row = []
        for attr in model.domain.attributes:
            if attr.name not in input_features:
                raise HTTPException(status_code=400, detail=f"Missing required feature: {attr.name}")
            
            val = input_features[attr.name]
            # Convert to float for continuous (Orange models generally take float representations natively)
            # For discrete, if they passed the string value, we need to map to index
            if attr.is_discrete and isinstance(val, str):
                 try:
                     val = attr.values.index(val)
                 except ValueError:
                     raise HTTPException(status_code=400, detail=f"Invalid value '{val}' for discrete feature '{attr.name}'. Expected one of: {attr.values}")
            data_row.append(float(val))
            
        # Reshape for single instance prediction
        X = [data_row]
        
        # Predict Class
        pred_idx = model(X)[0]
        predicted_class = model.domain.class_var.values[int(pred_idx)]
        
        # Predict Probabilities
        probs = model(X, model.Probs)[0]
        prob_dict = {
            model.domain.class_var.values[i]: round(float(probs[i]), 4)
            for i in range(len(model.domain.class_var.values))
        }
        
        return JSONResponse(content={
            "prediction": predicted_class,
            "probabilities": prob_dict
        })
        
    except HTTPException:
        raise
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
