import numpy as np
import os
 
 
def load_centerline(circuit_name: str) -> np.ndarray:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "Circuits", f"{circuit_name}.csv")
 
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No CSV found for circuit '{circuit_name}'. Expected at: {csv_path}")
 
    data = np.loadtxt(csv_path, delimiter=",") * 0.1
 
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"CSV must have exactly 2 columns (x, y). Got shape: {data.shape}")
 
    return data
