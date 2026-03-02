from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import pickle
import numpy as np
import Orange
import os
import json

def get_base_var(attr, depth=0):
    """
    Recursively extracts the original raw domain variable from a preprocessed
    Orange Attribute (which might be wrapped in ReplaceUnknowns, Indicator, etc.)
    """
    if hasattr(attr, 'compute_value') and attr.compute_value is not None:
        cv = attr.compute_value
        
        # Continuous indicators from discrete variables often use Indicator
        if type(cv).__name__ == 'Indicator' and hasattr(cv, 'variable'):
            return cv.variable
        
        # For ReplaceUnknowns or other wrappers that preserve the base variable
        if hasattr(cv, 'variable'):
             return get_base_var(cv.variable, depth + 1)
             
    return attr

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
            
        # Extract base RAW features (attributes) instead of the one-hot preprocessed ones
        raw_vars = set()
        for attr in model.domain.attributes:
            raw_vars.add(get_base_var(attr))

        features = []
        for attr in raw_vars:
            feature_info = {
                "name": attr.name,
                "type": "continuous" if type(attr).__name__ == "ContinuousVariable" else ("discrete" if type(attr).__name__ == "DiscreteVariable" else "other")
            }
            if hasattr(attr, "values"):
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
    
    # Reconstruct the raw domain expectations based on the base variables
    try:
        raw_vars = []
        seen = set()
        for attr in model.domain.attributes:
            base = get_base_var(attr)
            if base.name not in seen:
                raw_vars.append(base)
                seen.add(base.name)
            
        # Create an Orange Domain of raw inputs so that Orange logic can auto-compute transformations
        raw_domain = Orange.data.Domain(raw_vars, model.domain.class_var)
        
        data_row = []
        for attr in raw_domain.attributes:
            if attr.name not in input_features:
                raise HTTPException(status_code=400, detail=f"Missing required raw feature: {attr.name}")
            
            val = input_features[attr.name]
            
            # Convert continuous to float
            if type(attr).__name__ == "ContinuousVariable":
                try:
                    val = float(val)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid value '{val}' for continuous feature '{attr.name}'")
                    
            # For discrete variables, Orange Instance creation requires the integer index of the value
            if type(attr).__name__ == "DiscreteVariable" and isinstance(val, str):
                if val not in attr.values:
                    raise HTTPException(status_code=400, detail=f"Invalid value '{val}' for discrete feature '{attr.name}'. Expected one of: {attr.values}")
                val = float(attr.values.index(val))

            data_row.append(val)
            
        # Append the unknown class variable mapping
        # We need this because Domain expects [features..., class_var]. 
        # All underlying Orange Instance data must be floats, so we use float('nan') for unknown.
        import math
        data_row.append(math.nan)
            
        # Create a single Instance using the RAW domain.
        # When passed into the model, the model will inherently invoke the compute_values!
        instance = Orange.data.Instance(raw_domain, data_row)
        
        # Predict Class
        pred_idx = model(instance)
        predicted_class = model.domain.class_var.values[int(pred_idx)]
        
        # Predict Probabilities
        probs = model(instance, model.Probs)
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
         import traceback
         traceback.print_exc()
         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
