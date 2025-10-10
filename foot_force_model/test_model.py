import pandas as pd
import torch
import numpy as np
from foot_model import FootForceCNN

# Load model
model = FootForceCNN()
model.load_state_dict(torch.load("models/foot_force_classifier.pth"))
model.eval()

# Load a new CSV
df = pd.read_csv("./training_data/Push/foot_force_9_1759432297683.csv", header=None, names=["time", "foot1", "foot2", "foot3", "foot4"])
x = df[["foot1", "foot2", "foot3", "foot4"]].values.astype("float32")

# Ensure shape is correct (pad/trim to 90 like in your dataset)
target_len = 90
if len(x) < target_len:
    pad = np.repeat(x[-1][None, :], target_len - len(x), axis=0)
    x = np.concatenate([x, pad], axis=0)
elif len(x) > target_len:
    x = x[:target_len]

# Prepare tensor
x = torch.tensor(x).unsqueeze(0)  # shape: (1, 90, 4)

# Predict
with torch.no_grad():
    outputs = model(x)
    predicted = torch.argmax(outputs, dim=1).item()

# Map back to label
label_map = {0: "Null", 1: "Pull", 2: "Push", 3: "StrafeL", 4: "StrafeR"}  # adjust order to match training
print("Predicted:", label_map[predicted])
