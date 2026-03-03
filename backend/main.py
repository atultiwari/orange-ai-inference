from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import pickle
import numpy as np
import Orange
import os
import json
import io
import pandas as pd
import math

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

@app.post("/api/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Parses an uploaded CSV file and returns its columns and unique values for discrete features.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        columns_info = []
        for col in df.columns:
            col_type = "continuous"
            unique_vals = []
            
            # If dtype is object or category or boolean, treat as discrete
            if df[col].dtype == 'object' or df[col].dtype.name == 'category' or df[col].dtype == 'bool':
                col_type = "discrete"
                # Get unique values, drop NaNs, convert to string
                unique_vals = [str(v) for v in df[col].dropna().unique().tolist()][:100] # Limit to 100
                
            columns_info.append({
                "name": col,
                "type": col_type,
                "unique_values": unique_vals if col_type == "discrete" else []
            })
            
        return {
            "filename": file.filename,
            "columns": columns_info,
            "total_rows": len(df),
            "message": "CSV loaded and analyzed successfully."
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")

@app.post("/api/predict_batch")
async def predict_batch(
    file: UploadFile = File(...),
    filename: str = Form(..., description="The model filename (.pkcls)"),
    column_mapping: str = Form(..., description="JSON string mapping model features -> CSV columns"),
    value_mapping: str = Form(..., description="JSON string mapping CSV discrete values -> Model discrete values")
):
    """
    Executes a batch prediction on an uploaded CSV using a pre-uploaded model and a column/value mapping.
    Returns the annotated CSV as a downloadable file.
    """
    try:
        col_map = json.loads(column_mapping)
        val_map = json.loads(value_mapping)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON format for mapping arguments.")
        
    model_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found. Please upload it first.")
        
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        # Reconstruct raw domain
        raw_vars = []
        seen = set()
        for attr in model.domain.attributes:
            base = get_base_var(attr)
            if base.name not in seen:
                raw_vars.append(base)
                seen.add(base.name)
                
        raw_domain = Orange.data.Domain(raw_vars, model.domain.class_var)
        class_values = model.domain.class_var.values
        
        # Load CSV into Pandas
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        predictions = []
        probabilities = []
        
        for index, row in df.iterrows():
            data_row = []
            # Map each attribute
            for attr in raw_domain.attributes:
                model_feat = attr.name
                if model_feat not in col_map:
                    data_row.append(math.nan)
                    continue
                    
                csv_col = col_map[model_feat]
                if csv_col not in df.columns:
                    data_row.append(math.nan)
                    continue
                    
                raw_val = row[csv_col]
                if pd.isna(raw_val):
                    data_row.append(math.nan)
                    continue
                    
                if type(attr).__name__ == "ContinuousVariable":
                    try:
                        data_row.append(float(raw_val))
                    except ValueError:
                        data_row.append(math.nan)
                elif type(attr).__name__ == "DiscreteVariable":
                    raw_val_str = str(raw_val)
                    
                    # Apply value mapping if defined
                    resolved_val = raw_val_str
                    if model_feat in val_map and raw_val_str in val_map[model_feat]:
                        resolved_val = val_map[model_feat][raw_val_str]
                        
                    if resolved_val in attr.values:
                        data_row.append(float(attr.values.index(resolved_val)))
                    else:
                        data_row.append(math.nan)
                        
            # Append target
            data_row.append(math.nan)
            
            # Predict
            instance = Orange.data.Instance(raw_domain, data_row)
            try:
                pred_idx = model(instance)
                predicted_class = class_values[int(pred_idx)]
                probs = model(instance, model.Probs)
                
                predictions.append(predicted_class)
                probabilities.append(round(float(max(probs)), 4))
            except Exception as e:
                predictions.append("Error")
                probabilities.append(0.0)
                
        # Append results to the original dataframe
        df['Predicted_Class'] = predictions
        df['Confidence'] = probabilities
        
        # Stream response
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)
        
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=annotated_{file.filename.split('.')[0]}.csv"
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
