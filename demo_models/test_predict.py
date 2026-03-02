import pickle
import numpy as np
import Orange

with open('demo_01.pkcls', 'rb') as f:
    model = pickle.load(f)

print(f"Model domain: {model.domain}")
print(f"Attributes: {[attr.name for attr in model.domain.attributes]}")
print(f"Class var: {model.domain.class_var.name}")
print(f"Class var values: {model.domain.class_var.values}")

# Create a data table or instance for prediction
test_data = [[5.1, 3.5, 1.4, 0.2]]
# we can predict by passing a list or numpy array... let's check what Orange models expect.
try:
    pred = model(test_data)
    print(f"Prediction: {pred}")
    class_idx = int(pred[0])
    print(f"Class: {model.domain.class_var.values[class_idx]}")
except Exception as e:
    print(f"Error predicting with list: {e}")

try:
    pred_prob = model(test_data, model.Probs)
    print(f"Prediction probabilities: {pred_prob}")
except Exception as e:
    print(f"Error getting probabilities: {e}")

